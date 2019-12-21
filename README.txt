
python encode_image.py   --src_im sample.png --iteration 500

python image_morphing.py --latent_file1 latent_W/0.npy --latent_file2 latent_W/sample.npy

python make_morphgif.py

python image_crossover.py --src_im1  source_image/joker.png --src_im2  source_image/0.png --mask source_image/Blur_mask.png --iteration 1500

python facial_exchange.py --src_im1  source_image/sample.png --src_im2  source_image/0.png  --iteration 500

python semantic_edit.py --latent_file  latent_W/0.npy
