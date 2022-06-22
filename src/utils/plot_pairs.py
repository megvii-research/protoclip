import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap


def plot_pairs(imgs, texts,  suptitle, file_name='test.png', sample_per_row = 32):
        # imgs: a list of PIL-opened images
        # texts: a list of str, will be showed as title (per subplot)
        # suptitle: figure super title

    if len(imgs)<sample_per_row:
        row,column=1,len(imgs)
    else:
        row = (sample_per_row+len(imgs)-1)//sample_per_row
        column=sample_per_row
    
    plt.figure(figsize=(2*column,2.5*row))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.rc('font', size=10) 

    for i in range(len(imgs)):
        image=imgs[i]
        text=texts[i].replace('$','')
        if len(text)>40:
            text = text[:40]+'...'
        text = textwrap.fill(text, width=20)
        
        plt.subplot(row,column, i+1)
        plt.imshow(image)
        plt.text(
            x=int(image.size[0]/2),y=image.size[1]+30,s=text,
            fontsize=11,  va='top', ha='center',  
            bbox={'facecolor': 'white', 'edgecolor':'white',  'pad': 4,}
            )

        plt.xticks([])
        plt.yticks([])

    if suptitle is not None:
        plt.suptitle(suptitle, size='x-large')

        
    plt.savefig(file_name,  bbox_inches='tight')
    plt.close()