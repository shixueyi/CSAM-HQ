import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(pred_logits, image=None, t=2, sigma_gamma=3, sigma_alpha=20, sigma_beta=5):
    """
    pred_logits: 模型的输出 Logits (未经过 Sigmoid), shape (H, W)
    image: 原始 RGB 图像, shape (H, W, 3)。如果无图像，传入 None 或全 0 矩阵。
    """
    if hasattr(pred_logits, 'cpu'):
        pred_logits = pred_logits.squeeze().cpu().numpy()
        
    H, W = pred_logits.shape[:2]
    
    # 1. 计算概率 (Sigmoid)
    prob = 1.0 / (1.0 + np.exp(-pred_logits))
    
    # 2. 构造 CRF 输入 (背景 vs 前景)
    probs = np.stack([1.0 - prob, prob], axis=0) # Shape: (2, H, W)
    
    # 3. 设置 CRF
    d = dcrf.DenseCRF2D(W, H, 2)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    
    # 4. Pairwise Potentials
    # A. 空间平滑项 (Gaussian) - 仅依赖空间位置，不需要原图，永远执行
    d.addPairwiseGaussian(sxy=(sigma_gamma, sigma_gamma), compat=3, 
                          kernel=dcrf.DIAG_KERNEL, 
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # B. 外观一致项 (Bilateral) - 仅在提供了有效图像时执行
    if image is not None:
        if hasattr(image, 'cpu'):
            image = image.squeeze().cpu().numpy().astype(np.uint8)
            
        # 检查是否为全黑的占位图像
        if image.max() > 0:
            d.addPairwiseBilateral(sxy=(sigma_alpha, sigma_alpha), 
                                   srgb=(sigma_beta, sigma_beta, sigma_beta), 
                                   rgbim=image, 
                                   compat=3, 
                                   kernel=dcrf.DIAG_KERNEL, 
                                   normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # 5. 推理
    Q = d.inference(t)
    final_prob = np.array(Q)[1, :].reshape((H, W))
    
    return final_prob