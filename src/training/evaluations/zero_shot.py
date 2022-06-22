
import logging
import tqdm
import numpy as np
import faiss

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, STL10

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from open_clip import tokenize as clip_tokenizer


def zero_shot_classifier(model, classnames, templates, tokenizer, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates] 
            if tokenizer is None:
                texts = clip_tokenizer(texts).to(args.device) #tokenize
            else:
                texts = torch.stack([tokenizer(text) for text in texts]).to(args.device)
            
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding.cpu())
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)#.to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        all_image_features = []
        all_labels = []
        for images, target in tqdm.tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)

            if args.distributed and not args.horovod:
                image_features = model.module.encode_image(images)
            else:
                image_features = model.encode_image(images)

            image_features = F.normalize(image_features, dim=-1)
            image_features = image_features.detach().cpu()

            all_image_features.append(image_features)
            all_labels.append(target)
            
            logits = 100. * image_features @ classifier

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = 100.0 * (top1 / n)
    top5 = 100.0 * (top5 / n)
    
    all_image_features = torch.cat(all_image_features)
    all_labels = torch.cat(all_labels).numpy()
    
    
    return top1, top5, all_image_features, all_labels

def clustering_evaluation(features, labels):

    features = features.numpy().astype(np.float32)
    kmeans = faiss.Kmeans(
            d=features.shape[1], 
            k=int(max(labels)+1), 
            niter=100, 
            nredo=5,
            verbose=False, 
            gpu=True)
    kmeans.train(features)

    distance, img_plabels = kmeans.index.search(features, 1)
    img_plabels = np.array(img_plabels)
    img_plabels = np.reshape(img_plabels, img_plabels.shape[0])

    ari = adjusted_rand_score(img_plabels, labels)
    ami = adjusted_mutual_info_score(img_plabels, labels)

    return ari, ami


def zero_shot_eval(model, zeroshot_dataset, epoch, preprocess, tokenizer, args):

    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if zeroshot_dataset=='cifar10':
        dataset = CIFAR10(root=args.eval_data_dir, download=True, train=False, transform=preprocess)
    elif zeroshot_dataset=='cifar100':
        dataset = CIFAR100(root=args.eval_data_dir, download=True, train=False, transform=preprocess)
    elif zeroshot_dataset=='stl10':
        dataset = STL10(root=args.eval_data_dir, download=True, split='test', transform=preprocess)
    else:
        # for ['birdsnap', 'country211', 'flowers102', 'gtsrb', 'stanford_cars', 'ucf101']
        data_path = f'{args.eval_data_dir}/{zeroshot_dataset}/test'
        if zeroshot_dataset == 'ucf101':
            data_path += 'list01'
        logging.info(f'Loading data from  {data_path}')

        dataset = torchvision.datasets.ImageFolder(data_path, transform=preprocess)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)


    logging.info(f'Calculating text classifier for {zeroshot_dataset}')
    classnames, prompt_templates = get_class_names_and_templets[zeroshot_dataset]
    import copy
    classnames = copy.deepcopy(classnames)
    if zeroshot_dataset == 'birdsnap':
        # https://github.com/ml-jku/cloob/issues/10
        # FileNotFoundError: Found no valid file for the classes 046, 066, 123, 299, 302, 351, 403, 436, 465
        # these empty folders are removed
        empty_indexs = [46, 66, 123, 299, 302, 351, 403, 436, 465]
        for empty_index in empty_indexs[::-1]:
            del classnames[empty_index]
    classifier = zero_shot_classifier(model, classnames, prompt_templates, tokenizer, args)

    logging.info(f'Calculating image features for {zeroshot_dataset}')
    results = {}
    top1, top5, features, labels = run(model, classifier, dataloader, args)
    ari, ami = clustering_evaluation(features, labels)
    logging.info(f'{zeroshot_dataset} zero-shot accuracy: {top1:.2f}% (top5: {top5:.2f}%)')
    logging.info(f'{zeroshot_dataset} clustering evaluation: ARI: {ari:.4f}, AMI:{ami:.4f}')


    results[f'{zeroshot_dataset}-zeroshot-accuracy-top1'] = top1
    results[f'{zeroshot_dataset}-zeroshot-accuracy-top5'] = top5
    results[f'{zeroshot_dataset}-adjusted-rand-index'] = ari
    results[f'{zeroshot_dataset}-adjusted-mutual-info'] = ami

    
    for key, item in results.items():
        results[key] = float(item)
    
    return results


stl10_classnames = [
    'airplane',
    'bird',
    'car',
    'cat',
    'deer',
    'dog',
    'horse',
    'monkey',
    'ship',
    'truck',
]

stl10_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

cifar10_classnames = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

cifar10_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar100_classnames = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]

cifar100_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

birdsnap_classnames = [
    "Coopers Hawk", "Northern Goshawk", "Sharp shinned Hawk", "Golden Eagle", "White tailed Hawk", "Zone tailed Hawk",
    "Red tailed Hawk", "Rough legged Hawk", "Red shouldered Hawk", "Broad winged Hawk", "Ferruginous Hawk",
    "Swainsons Hawk", "Common Black Hawk", "Northern Harrier", "Swallow tailed Kite", "White tailed Kite", "Bald Eagle",
    "Mississippi Kite", "Harriss Hawk", "Snail Kite", "Bushtit", "Horned Lark", "Belted Kingfisher", "Pigeon Guillemot",
    "Black Guillemot", "Common Murre", "Northern Pintail", "American Wigeon", "Green winged Teal", "Cinnamon Teal",
    "Blue winged Teal", "Mottled Duck", "Eurasian Wigeon", "Mallard", "American Black Duck", "Gadwall", "Lesser Scaup",
    "Redhead", "Ring necked Duck", "Greater Scaup", "Canvasback", "Bufflehead", "Common Goldeneye", "Barrows Goldeneye",
    "Muscovy Duck", "Long tailed Duck", "Harlequin Duck", "Hooded Merganser", "Black Scoter", "White winged Scoter",
    "Surf Scoter", "Common Merganser", "Red breasted Merganser", "Ruddy Duck", "Common Eider",
    "Greater White fronted Goose", "Brant", "Canada Goose", "Cackling Goose", "Snow Goose", "Rosss Goose",
    "Trumpeter Swan", "Tundra Swan", "Mute Swan", "Fulvous Whistling Duck", "Anhinga", "White throated Swift",
    "Chimney Swift", "Limpkin", "Great Egret", "Great Blue Heron", "American Bittern", "Cattle Egret",
    "Little Blue Heron", "Reddish Egret", "Snowy Egret", "Tricolored Heron", "Least Bittern",
    "Yellow crowned Night Heron", "Black crowned Night Heron", "Cedar Waxwing", "Bohemian Waxwing", "Lapland Longspur",
    "Chestnut collared Longspur", "Snow Bunting", "Lesser Nighthawk", "Common Nighthawk", "Northern Cardinal",
    "Pyrrhuloxia", "Lazuli Bunting", "Blue Grosbeak", "Painted Bunting", "Indigo Bunting", "Rose breasted Grosbeak",
    "Black headed Grosbeak", "Hepatic Tanager", "Western Tanager", "Scarlet Tanager", "Summer Tanager", "Dickcissel",
    "Turkey Vulture", "Black Vulture", "Brown Creeper", "Piping Plover", "Mountain Plover", "Snowy Plover",
    "Semipalmated Plover", "Killdeer", "Wilsons Plover", "American Golden Plover", "Pacific Golden Plover",
    "Black bellied Plover", "Wood Stork", "American Dipper", "Rock Pigeon", "Inca Dove", "Common Ground Dove",
    "Band tailed Pigeon", "Eurasian Collared Dove", "White winged Dove", "Mourning Dove", "Western Scrub Jay",
    "Florida Scrub Jay", "Mexican Jay", "American Crow", "Common Raven", "Chihuahuan Raven", "Fish Crow", "Blue Jay",
    "Stellers Jay", "Green Jay", "Clarks Nutcracker", "Gray Jay", "Black billed Magpie", "Yellow billed Magpie",
    "Groove billed Ani", "Yellow billed Cuckoo", "Black billed Cuckoo", "Greater Roadrunner", "Rufous crowned Sparrow",
    "Saltmarsh Sparrow", "Henslows Sparrow", "Le Contes Sparrow", "Seaside Sparrow", "Nelsons Sparrow",
    "Grasshopper Sparrow", "Black throated Sparrow", "Olive Sparrow", "Lark Bunting", "Lark Sparrow", "Dark eyed Junco",
    "Yellow eyed Junco", "Swamp Sparrow", "Lincolns Sparrow", "Song Sparrow", "California Towhee", "Canyon Towhee",
    "Fox Sparrow", "Cassins Sparrow", "Green tailed Towhee", "Eastern Towhee", "Spotted Towhee", "Vesper Sparrow",
    "American Tree Sparrow", "Black chinned Sparrow", "Brewers Sparrow", "Clay colored Sparrow", "Chipping Sparrow",
    "Field Sparrow", "White throated Sparrow", "Golden crowned Sparrow", "White crowned Sparrow", "Harriss Sparrow",
    "Crested Caracara", "Merlin", "Prairie Falcon", "Peregrine Falcon", "American Kestrel", "Magnificent Frigatebird",
    "Common Redpoll", "Hoary Redpoll", "Evening Grosbeak", "Cassins Finch", "House Finch", "Purple Finch",
    "Black Rosy Finch", "Brown capped Rosy Finch", "Gray crowned Rosy Finch", "Red Crossbill", "White winged Crossbill",
    "Pine Grosbeak", "Pine Siskin", "Lesser Goldfinch", "American Goldfinch", "Common Loon", "Pacific Loon",
    "Red throated Loon", "Sandhill Crane", "Black Oystercatcher", "American Oystercatcher", "Barn Swallow",
    "Cave Swallow", "Cliff Swallow", "Purple Martin", "Northern Rough winged Swallow", "Tree Swallow",
    "Violet green Swallow", "Red winged Blackbird", "Bobolink", "Rusty Blackbird", "Brewers Blackbird",
    "Bullocks Oriole", "Hooded Oriole", "Baltimore Oriole", "Audubons Oriole", "Altamira Oriole", "Scotts Oriole",
    "Orchard Oriole", "Bronzed Cowbird", "Brown headed Cowbird", "Boat tailed Grackle", "Great tailed Grackle",
    "Common Grackle", "Eastern Meadowlark", "Western Meadowlark", "Yellow headed Blackbird", "Northern Shrike",
    "Loggerhead Shrike", "Bonapartes Gull", "Herring Gull", "California Gull", "Mew Gull", "Ring billed Gull",
    "Glaucous winged Gull", "Iceland Gull", "Heermanns Gull", "Glaucous Gull", "Great Black backed Gull",
    "Western Gull", "Thayers Gull", "Laughing Gull", "Franklins Gull", "Black legged Kittiwake", "Black Skimmer",
    "Black Tern", "Gull billed Tern", "Caspian Tern", "Roseate Tern", "Forsters Tern", "Common Tern", "Arctic Tern",
    "Least Tern", "Royal Tern", "Sandwich Tern", "Gray Catbird", "Northern Mockingbird", "Sage Thrasher",
    "Curve billed Thrasher", "Long billed Thrasher", "California Thrasher", "Brown Thrasher", "American Pipit",
    "California Quail", "Gambels Quail", "Scaled Quail", "Northern Bobwhite", "Osprey", "Black crested Titmouse",
    "Tufted Titmouse", "Oak Titmouse", "Bridled Titmouse", "Black capped Chickadee", "Carolina Chickadee",
    "Mountain Chickadee", "Boreal Chickadee", "Chestnut backed Chickadee", "Canada Warbler", "Wilsons Warbler",
    "Mourning Warbler", "Common Yellowthroat", "Worm eating Warbler", "Yellow breasted Chat", "Black and white Warbler",
    "Painted Redstart", "Connecticut Warbler", "Orange crowned Warbler", "Tennessee Warbler", "Nashville Warbler",
    "Louisiana Waterthrush", "Northern Waterthrush", "Prothonotary Warbler", "Ovenbird", "Northern Parula",
    "Black throated Blue Warbler", "Bay breasted Warbler", "Cerulean Warbler", "Hooded Warbler",
    "Yellow rumped Warbler", "Prairie Warbler", "Yellow throated Warbler", "Blackburnian Warbler", "Magnolia Warbler",
    "Black throated Gray Warbler", "Palm Warbler", "Chestnut sided Warbler", "Yellow Warbler", "Pine Warbler",
    "American Redstart", "Blackpoll Warbler", "Cape May Warbler", "Townsends Warbler", "Black throated Green Warbler",
    "Golden winged Warbler", "Blue winged Warbler", "House Sparrow", "American White Pelican", "Brown Pelican",
    "Double crested Cormorant", "Neotropic Cormorant", "Great Cormorant", "Brandts Cormorant", "Wild Turkey",
    "Ring necked Pheasant", "Ruffed Grouse", "Greater Sage Grouse", "Sooty Grouse", "Dusky Grouse", "Spruce Grouse",
    "Willow Ptarmigan", "White tailed Ptarmigan", "Rock Ptarmigan", "Greater Prairie Chicken", "Sharp tailed Grouse",
    "Northern Flicker", "Pileated Woodpecker", "Golden fronted Woodpecker", "Red bellied Woodpecker",
    "Red headed Woodpecker", "Acorn Woodpecker", "Lewiss Woodpecker", "Gila Woodpecker", "White headed Woodpecker",
    "Black backed Woodpecker", "Red cockaded Woodpecker", "American Three toed Woodpecker", "Nuttalls Woodpecker",
    "Downy Woodpecker", "Ladder backed Woodpecker", "Hairy Woodpecker", "Red naped Sapsucker", "Red breasted Sapsucker",
    "Williamsons Sapsucker", "Yellow bellied Sapsucker", "Clarks Grebe", "Western Grebe", "Horned Grebe",
    "Red necked Grebe", "Eared Grebe", "Pied billed Grebe", "Least Grebe", "Blue gray Gnatcatcher", "Monk Parakeet",
    "Phainopepla", "American Coot", "Common Gallinule", "Purple Gallinule", "Sora", "King Rail", "Virginia Rail",
    "Clapper Rail", "Black necked Stilt", "American Avocet", "Ruby crowned Kinglet", "Golden crowned Kinglet", "Verdin",
    "Red Phalarope", "Red necked Phalarope", "Wilsons Phalarope", "Spotted Sandpiper", "Surfbird", "Ruddy Turnstone",
    "Black Turnstone", "Upland Sandpiper", "Sanderling", "Dunlin", "Bairds Sandpiper", "Red Knot",
    "White rumped Sandpiper", "Stilt Sandpiper", "Purple Sandpiper", "Western Sandpiper", "Pectoral Sandpiper",
    "Least Sandpiper", "Rock Sandpiper", "Semipalmated Sandpiper", "Wilsons Snipe", "Short billed Dowitcher",
    "Long billed Dowitcher", "Marbled Godwit", "Long billed Curlew", "Whimbrel", "American Woodcock",
    "Lesser Yellowlegs", "Wandering Tattler", "Greater Yellowlegs", "Willet", "Solitary Sandpiper",
    "Red breasted Nuthatch", "White breasted Nuthatch", "Brown headed Nuthatch", "Pygmy Nuthatch",
    "Northern Saw whet Owl", "Short eared Owl", "Long eared Owl", "Burrowing Owl", "Snowy Owl", "Great Horned Owl",
    "Ferruginous Pygmy Owl", "Eastern Screech Owl", "Western Screech Owl", "Elf Owl", "Great Gray Owl", "Spotted Owl",
    "Barred Owl", "Northern Hawk Owl", "European Starling", "Northern Gannet", "Wrentit", "White Ibis",
    "White faced Ibis", "Glossy Ibis", "Black chinned Hummingbird", "Ruby throated Hummingbird", "Annas Hummingbird",
    "Costas Hummingbird", "Broad billed Hummingbird", "Calliope Hummingbird", "Broad tailed Hummingbird",
    "Rufous Hummingbird", "Allens Hummingbird", "Cactus Wren", "Canyon Wren", "Marsh Wren", "Sedge Wren", "Rock Wren",
    "Bewicks Wren", "Carolina Wren", "House Wren", "Winter Wren", "Pacific Wren", "Elegant Trogon", "Veery",
    "Hermit Thrush", "Gray cheeked Thrush", "Swainsons Thrush", "Wood Thrush", "Varied Thrush", "Townsends Solitaire",
    "Mountain Bluebird", "Western Bluebird", "Eastern Bluebird", "American Robin", "Olive sided Flycatcher",
    "Western Wood Pewee", "Eastern Wood Pewee", "Alder Flycatcher", "Pacific slope Flycatcher",
    "Yellow bellied Flycatcher", "Hammonds Flycatcher", "Least Flycatcher", "Dusky Flycatcher",
    "Cordilleran Flycatcher", "Willow Flycatcher", "Acadian Flycatcher", "Gray Flycatcher", "Vermilion Flycatcher",
    "Black Phoebe", "Eastern Phoebe", "Says Phoebe", "Ash throated Flycatcher", "Great Crested Flycatcher",
    "Brown crested Flycatcher", "Great Kiskadee", "Couchs Kingbird", "Gray Kingbird", "Scissor tailed Flycatcher",
    "Tropical Kingbird", "Eastern Kingbird", "Western Kingbird", "Cassins Kingbird", "Bells Vireo", "Cassins Vireo",
    "Yellow throated Vireo", "Warbling Vireo", "White eyed Vireo", "Huttons Vireo", "Red eyed Vireo",
    "Philadelphia Vireo", "Plumbeous Vireo", "Blue headed Vireo"
]

country211_classnames = [
    'Andorra', 'United Arab Emirates', 'Afghanistan', 'Antigua and Barbuda', 'Anguilla', 'Albania', 'Armenia', 'Angola',
    'Antarctica', 'Argentina', 'Austria', 'Australia', 'Aruba', 'Aland Islands', 'Azerbaijan', 'Bosnia and Herzegovina',
    'Barbados', 'Bangladesh', 'Belgium', 'Burkina Faso', 'Bulgaria', 'Bahrain', 'Benin', 'Bermuda', 'Brunei Darussalam',
    'Bolivia', 'Bonaire, Saint Eustatius and Saba', 'Brazil', 'Bahamas', 'Bhutan', 'Botswana', 'Belarus', 'Belize',
    'Canada', 'DR Congo', 'Central African Republic', 'Switzerland', "Cote d'Ivoire", 'Cook Islands', 'Chile',
    'Cameroon', 'China', 'Colombia', 'Costa Rica', 'Cuba', 'Cabo Verde', 'Curacao', 'Cyprus', 'Czech Republic',
    'Germany', 'Denmark', 'Dominica', 'Dominican Republic', 'Algeria', 'Ecuador', 'Estonia', 'Egypt', 'Spain',
    'Ethiopia', 'Finland', 'Fiji', 'Falkland Islands', 'Faeroe Islands', 'France', 'Gabon', 'United Kingdom', 'Grenada',
    'Georgia', 'French Guiana', 'Guernsey', 'Ghana', 'Gibraltar', 'Greenland', 'Gambia', 'Guadeloupe', 'Greece',
    'South Georgia and South Sandwich Is.', 'Guatemala', 'Guam', 'Guyana', 'Hong Kong', 'Honduras', 'Croatia', 'Haiti',
    'Hungary', 'Indonesia', 'Ireland', 'Israel', 'Isle of Man', 'India', 'Iraq', 'Iran', 'Iceland', 'Italy', 'Jersey',
    'Jamaica', 'Jordan', 'Japan', 'Kenya', 'Kyrgyz Republic', 'Cambodia', 'St. Kitts and Nevis', 'North Korea',
    'South Korea', 'Kuwait', 'Cayman Islands', 'Kazakhstan', 'Laos', 'Lebanon', 'St. Lucia', 'Liechtenstein',
    'Sri Lanka', 'Liberia', 'Lithuania', 'Luxembourg', 'Latvia', 'Libya', 'Morocco', 'Monaco', 'Moldova', 'Montenegro',
    'Saint-Martin', 'Madagascar', 'Macedonia', 'Mali', 'Myanmar', 'Mongolia', 'Macau', 'Martinique', 'Mauritania',
    'Malta', 'Mauritius', 'Maldives', 'Malawi', 'Mexico', 'Malaysia', 'Mozambique', 'Namibia', 'New Caledonia',
    'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New Zealand', 'Oman', 'Panama', 'Peru',
    'French Polynesia', 'Papua New Guinea', 'Philippines', 'Pakistan', 'Poland', 'Puerto Rico', 'Palestine', 'Portugal',
    'Palau', 'Paraguay', 'Qatar', 'Reunion', 'Romania', 'Serbia', 'Russia', 'Rwanda', 'Saudi Arabia', 'Solomon Islands',
    'Seychelles', 'Sudan', 'Sweden', 'Singapore', 'St. Helena', 'Slovenia', 'Svalbard and Jan Mayen Islands',
    'Slovakia', 'Sierra Leone', 'San Marino', 'Senegal', 'Somalia', 'South Sudan', 'El Salvador', 'Sint Maarten',
    'Syria', 'Eswatini', 'Togo', 'Thailand', 'Tajikistan', 'Timor-Leste', 'Turkmenistan', 'Tunisia', 'Tonga', 'Turkey',
    'Trinidad and Tobago', 'Taiwan', 'Tanzania', 'Ukraine', 'Uganda', 'United States', 'Uruguay', 'Uzbekistan',
    'Vatican', 'Venezuela', 'British Virgin Islands', 'United States Virgin Islands', 'Vietnam', 'Vanuatu', 'Samoa',
    'Kosovo', 'Yemen', 'South Africa', 'Zambia', 'Zimbabwe',
]

flowers102_classnames = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily',
    'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea',
    'spear thistle', 'yellow iris', 'globe flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth',
    'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation',
    'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
    'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia',
    'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink and yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush',
    'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium',
    'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen',
    'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'air plant', 'foxglove', 'bougainvillea', 'camellia',
    'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily',
]

gtsrb_classnames = [
    'red and white circle 20 kph speed limit', 'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit', 'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit', 'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit', 'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit', 'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing', 'red and white triangle road intersection warning',
    'white and yellow diamond priority road', 'red and white upside down triangle yield right-of-way', 'stop',
    'empty red and white circle', 'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry', 'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning', 'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning', 'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning', 'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit', 'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory', 'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory', 'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory', 'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]

imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster",
    "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul",
    "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl",
    "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
    "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama",
    "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon",
    "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake",
    "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor",
    "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper",
    "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion",
    "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail",
    "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater",
    "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker",
    "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm",
    "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod",
    "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird",
    "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin",
    "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale",
    "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu",
    "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound",
    "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound",
    "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
    "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier",
    "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
    "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
    "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever",
    "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter",
    "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois",
    "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie",
    "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer",
    "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
    "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote",
    "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat",
    "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar",
    "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle",
    "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis",
    "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish",
    "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel",
    "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus",
    "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest",
    "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla",
    "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur",
    "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey",
    "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant",
    "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish",
    "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion",
    "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle",
    "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon",
    "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn",
    "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass",
    "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse",
    "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
    "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette",
    "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence",
    "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet",
    "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store",
    "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane",
    "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk",
    "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick",
    "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center",
    "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck",
    "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen",
    "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown",
    "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray",
    "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive",
    "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt",
    "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans",
    "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle",
    "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter",
    "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass",
    "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover",
    "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith",
    "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped",
    "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse",
    "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer",
    "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace",
    "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio",
    "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum",
    "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow",
    "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot",
    "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck",
    "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope",
    "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control",
    "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick",
    "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale",
    "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine",
    "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine",
    "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero",
    "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle",
    "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit",
    "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt",
    "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine",
    "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch",
    "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano",
    "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct",
    "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig",
    "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool",
    "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign",
    "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple",
    "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough",
    "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble",
    "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano",
    "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus",
    "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

stanford_cars_classnames = [
    'AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008',
    'Acura TSX Sedan 2012', 'Acura Integra Type R 2001', 'Acura ZDX Hatchback 2012',
    'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012',
    'Aston Martin Virage Convertible 2012', 'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008',
    'Audi A5 Coupe 2012', 'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012', 'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994',
    'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011', 'Audi S5 Convertible 2012',
    'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012',
    'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012',
    'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007', 'BMW X5 SUV 2007',
    'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010', 'BMW M6 Convertible 2010', 'BMW X3 SUV 2012',
    'BMW Z4 Convertible 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009',
    'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007',
    'Bentley Continental Flying Spur Sedan 2007', 'Bugatti Veyron 16.4 Convertible 2009',
    'Bugatti Veyron 16.4 Coupe 2009', 'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012',
    'Buick Enclave SUV 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012',
    'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette ZR1 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012',
    'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007',
    'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Express Cargo Van 2007',
    'Chevrolet Avalanche Crew Cab 2012', 'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010',
    'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Express Van 2007',
    'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012', 'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010',
    'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008',
    'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012',
    'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997', 'Dodge Ram Pickup 3500 Crew Cab 2010',
    'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012',
    'Dodge Dakota Crew Cab 2010', 'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011',
    'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007', 'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009',
    'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari FF Coupe 2012',
    'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012',
    'Fisker Karma Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007',
    'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012', 'Ford Ranger SuperCab 2011',
    'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007',
    'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012',
    'GMC Yukon Hybrid SUV 2012', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993',
    'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009', 'Honda Odyssey Minivan 2012',
    'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012',
    'Hyundai Veloster Hatchback 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012',
    'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007',
    'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012',
    'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012', 'Infiniti G Coupe IPL 2012',
    'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012',
    'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012',
    'Lamborghini Reventon Coupe 2008', 'Lamborghini Aventador Coupe 2012',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001',
    'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012', 'Lincoln Town Car Sedan 2011',
    'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011',
    'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012',
    'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012',
    'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012',
    'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012', 'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999',
    'Porsche Panamera Sedan 2012', 'Ram C/V Cargo Van Minivan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Ghost Sedan 2012',
    'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009',
    'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012',
    'Tesla Model S Sedan 2012', 'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012',
    'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012', 'Volkswagen Golf Hatchback 1991',
    'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007',
    'smart fortwo Convertible 2012',
]

ucf101_classnames = [
    'Apply Eye Makeup', 'Apply Lipstick', 'Archery', 'Baby Crawling', 'Balance Beam', 'Band Marching', 'Baseball Pitch',
    'Basketball', 'Basketball Dunk', 'Bench Press', 'Biking', 'Billiards', 'Blow Dry Hair', 'Blowing Candles',
    'Body Weight Squats', 'Bowling', 'Boxing Punching Bag', 'Boxing Speed Bag', 'Breast Stroke', 'Brushing Teeth',
    'Clean And Jerk', 'Cliff Diving', 'Cricket Bowling', 'Cricket Shot', 'Cutting In Kitchen', 'Diving', 'Drumming',
    'Fencing', 'Field Hockey Penalty', 'Floor Gymnastics', 'Frisbee Catch', 'Front Crawl', 'Golf Swing', 'Haircut',
    'Hammer Throw', 'Hammering', 'Hand Stand Pushups', 'Handstand Walking', 'Head Massage', 'High Jump', 'Horse Race',
    'Horse Riding', 'Hula Hoop', 'Ice Dancing', 'Javelin Throw', 'Juggling Balls', 'Jump Rope', 'Jumping Jack',
    'Kayaking', 'Knitting', 'Long Jump', 'Lunges', 'Military Parade', 'Mixing', 'Mopping Floor', 'Nunchucks',
    'Parallel Bars', 'Pizza Tossing', 'Playing Cello', 'Playing Daf', 'Playing Dhol', 'Playing Flute', 'Playing Guitar',
    'Playing Piano', 'Playing Sitar', 'Playing Tabla', 'Playing Violin', 'Pole Vault', 'Pommel Horse', 'Pull Ups',
    'Punch', 'Push Ups', 'Rafting', 'Rock Climbing Indoor', 'Rope Climbing', 'Rowing', 'Salsa Spin', 'Shaving Beard',
    'Shotput', 'Skate Boarding', 'Skiing', 'Skijet', 'Sky Diving', 'Soccer Juggling', 'Soccer Penalty', 'Still Rings',
    'Sumo Wrestling', 'Surfing', 'Swing', 'Table Tennis Shot', 'Tai Chi', 'Tennis Swing', 'Throw Discus',
    'Trampoline Jumping', 'Typing', 'Uneven Bars', 'Volleyball Spiking', 'Walking With Dog', 'Wall Pushups',
    'Writing On Board', 'Yo Yo',
]

birdsnap_templates = [
    lambda c: f'a photo of a {c}, a type of bird.',
]

country211_templates = [
    lambda c: f'a photo i took in {c}.',
    lambda c: f'a photo i took while visiting {c}.',
    lambda c: f'a photo from my home country of {c}.',
    lambda c: f'a photo from my visit to {c}.',
    lambda c: f'a photo showing the country of {c}.',
]

flowers102_templates = [
    lambda c: f'a photo of a {c}, a type of flower.'
]

gtsrb_templates = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

imagenet_templates = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

stanford_cars_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

ucf101_templates = [
    lambda c: f'a photo of a person {c}.',
    lambda c: f'a video of a person {c}.',
    lambda c: f'a example of a person {c}.',
    lambda c: f'a demonstration of a person {c}.',
    lambda c: f'a photo of the person {c}.',
    lambda c: f'a video of the person {c}.',
    lambda c: f'a example of the person {c}.',
    lambda c: f'a demonstration of the person {c}.',
    lambda c: f'a photo of a person using {c}.',
    lambda c: f'a video of a person using {c}.',
    lambda c: f'a example of a person using {c}.',
    lambda c: f'a demonstration of a person using {c}.',
    lambda c: f'a photo of the person using {c}.',
    lambda c: f'a video of the person using {c}.',
    lambda c: f'a example of the person using {c}.',
    lambda c: f'a demonstration of the person using {c}.',
    lambda c: f'a photo of a person doing {c}.',
    lambda c: f'a video of a person doing {c}.',
    lambda c: f'a example of a person doing {c}.',
    lambda c: f'a demonstration of a person doing {c}.',
    lambda c: f'a photo of the person doing {c}.',
    lambda c: f'a video of the person doing {c}.',
    lambda c: f'a example of the person doing {c}.',
    lambda c: f'a demonstration of the person doing {c}.',
    lambda c: f'a photo of a person during {c}.',
    lambda c: f'a video of a person during {c}.',
    lambda c: f'a example of a person during {c}.',
    lambda c: f'a demonstration of a person during {c}.',
    lambda c: f'a photo of the person during {c}.',
    lambda c: f'a video of the person during {c}.',
    lambda c: f'a example of the person during {c}.',
    lambda c: f'a demonstration of the person during {c}.',
    lambda c: f'a photo of a person performing {c}.',
    lambda c: f'a video of a person performing {c}.',
    lambda c: f'a example of a person performing {c}.',
    lambda c: f'a demonstration of a person performing {c}.',
    lambda c: f'a photo of the person performing {c}.',
    lambda c: f'a video of the person performing {c}.',
    lambda c: f'a example of the person performing {c}.',
    lambda c: f'a demonstration of the person performing {c}.',
    lambda c: f'a photo of a person practicing {c}.',
    lambda c: f'a video of a person practicing {c}.',
    lambda c: f'a example of a person practicing {c}.',
    lambda c: f'a demonstration of a person practicing {c}.',
    lambda c: f'a photo of the person practicing {c}.',
    lambda c: f'a video of the person practicing {c}.',
    lambda c: f'a example of the person practicing {c}.',
    lambda c: f'a demonstration of the person practicing {c}.',
]



get_class_names_and_templets = {
    'birdsnap': (birdsnap_classnames, birdsnap_templates), 
    'country211': (country211_classnames, country211_templates), 
    'flowers102': (flowers102_classnames, flowers102_templates), 
    'gtsrb': (gtsrb_classnames, gtsrb_templates), 
    'stanford_cars': (stanford_cars_classnames, stanford_cars_templates),  
    'ucf101': (ucf101_classnames, ucf101_templates), 
    'imagenet': (imagenet_classnames, imagenet_templates),
    'stl10': (stl10_classnames, stl10_templates),
    'cifar10': (cifar10_classnames, cifar10_templates),
    'cifar100': (cifar100_classnames, cifar100_templates),
}