このリポジトリは以下の論文の非公式かつ部分的な実装です。

Abdal, R., Qin, Y., & Wonka, P. (2019). Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space? Retrieved from http://arxiv.org/abs/1904.03189 ↩

Abdal, R., Qin, Y., & Wonka, P. (2019). Image2StyleGAN++: How to Edit the Embedded Images? Retrieved from http://arxiv.org/abs/1911.11544 ↩

Yang, C., Lim, S.-N., & Ai, F. (n.d.). Unconstrained Facial Expression Transfer using Style-based Generator. https://arxiv.org/abs/1912.06253 ↩

Shen, Y., Gu, J., Tang, X., & Zhou, B. (2019). Interpreting the Latent Space of GANs for Semantic Face Editing. Retrieved from http://arxiv.org/abs/1907.10786 ↩

またこのコードはStyleGANの公式実装を使用しています。
https://github.com/NVlabs/stylegan

さらにpretrainedモデルをtensorflowからpytorchに移植する際、以下のリポジトリ内のコードを使用しています。
https://github.com/lernapparat/lernapparat

詳しくは以下のqiita記事を参照してください
https://qiita.com/pacifinapacific/private/1d6cca0ff4060e12d336

```bash
python encode_image.py   --src_im sample.png --iteration 500
```

```
python image_morphing.py --latent_file1 latent_W/0.npy --latent_file2 latent_W/sample.npy
```
```
python make_morphgif.py
```
```
python image_crossover.py --src_im1  source_image/joker.png --src_im2  source_image/0.png --mask source_image/Blur_mask.png --iteration 1500
```
```
python facial_exchange.py --src_im1  source_image/sample.png --src_im2  source_image/0.png  --iteration 500
```
```

python semantic_edit.py --latent_file  latent_W/0.npy
```
