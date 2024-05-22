# image_risk_control_system
基于我的个人认知搭建图片风险控制系统（视频等其他模态可参考）
## 图片检索系统
### 向量数据库
    选用milvus（因为开源），
        个人使用可以安装单实例https://milvus.io/docs/install_standalone-docker.md
        分布式部署or其他参考官网介绍，在python包装层是基本一致的    

    clip 选型：
        慎重，选型完决定了向量数据库的尺寸、后续的其他任务等
        中文clip https://modelscope.cn/models/iic/multi-modal_clip-vit-base-patch16_zh/summary
        英文clip可以直接用clip load
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    
## nsfw分类
    参考另一个项目 https://github.com/snowAIzxm/nsfw-clip
## ocr
    可参考优秀的开源项目，依据需求进行迭代or替换，以下是中文的ocr项目
    检测：https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary
    识别：https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary



