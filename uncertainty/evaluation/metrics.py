import numpy as np
from image_metrics import PSNRModule, SSIMModule, LPIPSModule

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def calculate_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)

def calculate_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)

def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def calculate_psnr(preds, target, mask=None):
    psnr_module = PSNRModule()
    return psnr_module(preds, target, mask)

def calculate_ssim(preds, target, mask=None):
    ssim_module = SSIMModule()
    return ssim_module(preds, target, mask)

def calculate_lpips(preds, target, mask=None):
    lpips_module = LPIPSModule()
    return lpips_module(preds, target, mask)
