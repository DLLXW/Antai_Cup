# Antai Cup-International E-commerce Artificial Intelligence Challenge　　

## 关于ncf_antai说明
    train_ncf.csv里面包含90k的数据（userID,itemID,timestamp,rating）四列，来自与100k的原始数据的去重处理。
    main.py里面定义了网络，main()函数里面是实现分块读取原始数据，并且利用该数据生成dataset用于train的逻辑。
           －　训练过程
           －　预测，评估过程。HR和NDCG两个指标
    dataset.py用于生成dataset，本demo的网络输入是(userID,itemID)。为了充分利用原始数据，可以将国家信息，商店信息，价格信息等
    都用于训练。
    

