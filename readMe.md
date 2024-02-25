## MVTEC AD


dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad/

- visualize data in embeddingspace on tensorboard with TSNE, UMAP and PCA
- Clip embeddings and Wideresnet50 embeddings
- create synthetic anomalies from testset anomalies as suggested in https://arxiv.org/abs/2202.08088 


![image](https://github.com/buehlpa/MVTec_Visualizer/assets/64488738/c1df57ea-cdea-4a57-b654-5c740dfc86f2)

navigate to the data directory and download & extract the data  with:

```
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz && tar -xf mvtec_anomaly_detection.tar.xz
```
### Funky Visualization with a Treemap
![image](https://github.com/buehlpa/MVTec_Visualizer/assets/64488738/563f0eff-1192-4667-aa31-e8114f1ea681)

### PCA with tensorboard 
![image](https://github.com/buehlpa/MVTec_Visualizer/assets/64488738/4e55d536-e8c2-4d26-8e69-3985e9e6f2d1)
