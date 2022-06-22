import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


# - - - - - - - - - - - - - - - - - - - - - - - - - - 


def assign_learning_rate_seperately(optimizer, lr_visual, lr_non_visual):
    optimizer.param_groups[0]["lr"] = lr_visual
    optimizer.param_groups[1]["lr"] = lr_visual
    optimizer.param_groups[2]["lr"] = lr_non_visual
    optimizer.param_groups[3]["lr"] = lr_non_visual

def protoclip_cosine_lr(optimizer, visual_base_lr, text_base_lr, warmup_length, total_steps, visual_steps, text_start_step, text_end_step):
    def _lr_adjuster(step):
        
        # get lr_visual for visual backbone
        if step < warmup_length:
            lr_visual = _warmup_lr(visual_base_lr, warmup_length, step)
        else:
            e_visual = step - warmup_length
            es_visual = visual_steps - warmup_length
            lr_visual = 0.5 * (1 + np.cos(np.pi * e_visual / es_visual)) * visual_base_lr
            if step > visual_steps:
                lr_visual = 0
        
        # get lr_non_visual for rest parameters
        if step < text_start_step:
            lr_non_visual = 0
        else:
            step -= text_start_step
            if step < warmup_length:
                lr_non_visual = _warmup_lr(text_base_lr, warmup_length, step)
            else:
                e_non_visual = step - warmup_length
                es_non_visual = text_end_step - text_start_step - warmup_length
                lr_non_visual = 0.5 * (1 + np.cos(np.pi * e_non_visual / es_non_visual)) * text_base_lr
                if step > text_end_step:
                    lr_non_visual = 0

        assign_learning_rate_seperately(optimizer, lr_visual, lr_non_visual)
        return [lr_visual, lr_non_visual]
    return _lr_adjuster
