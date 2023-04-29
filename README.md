# SolAI

## Booming Decentralized Solar Power in Africa’s Cities. Satellite Imagery and Deep Learning Provide Cutting-Edge Data on Electrification

![alt text](https://github.com/hugotmdd/solAI/blob/main/gitimage/background_pic.png)

This github repository provides a simplified version of the code used in the project titled "Booming Decentralized Solar Power in Africa’s Cities. Satellite Imagery and Deep Learning Provide Cutting-Edge Data on Electrification." You can find the original paper published by the French Institute of International Relations (IFRI) [here](https://www.ifri.org/en/publications/briefings-de-lifri/booming-decentralized-solar-power-africas-cities-satellite-imagery) (and in french [here](https://www.ifri.org/fr/publications/briefings-de-lifri/solaire-decentralise-lassaut-villes-africaines-une-analyse-originale)).

The project provides an analysis of the decentralized solar capacity installed in 14 Sub-Saharan African cities, using data obtained from millions of satellite images. The findings reveal that the decentralized solar capacity in these cities is between 184 MW and 231 MW, which represents almost 10% of the centralized solar capacity installed in the region, excluding South Africa. The analysis suggests that extending the study to other African cities could increase this estimate.

Moreover, after crossing this database with Afrobarometer's latest geolocalized survey, the study highlights that people in high-income households are more likely to adopt decentralized solar systems, irrespective of the reliability of their power supply from centralized grids. This trend implies a gradual shift in power dynamics from national power grids to customers. As such, it presents an existential challenge to the electricity sectors in Sub-Saharan Africa, as their most solvent customers adopt hybrid power supply strategies.

We hope that the code used in this project can serve as a valuable resource for researchers and practitioners interested in analyzing the solar capacity and electrification trends in Africa's urban areas. Therefore, we have made it available on this GitHub repository for easy access and use. Please feel free to explore the code and adapt it to your needs.

## Usage instructions

The code includes a classification and segmentation model. In this project, we have used Comet ML and Hydra to monitor and follow the training process.

To get started, please clone the repository using the command : 

```
git clone https://github.com/hugotmdd/solAI.git
```

and navigate to the "solAI" directory using :

```
cd solAI
```

You first need to add your training images in the folder /data/classification/images along with a csv comprising the image path and labels in /data/classification for the classification model, and in /data/segmentation/images and /data/segmentation/masks with a csv in /data/segmentation with their paths for the segmentation. You can also change the path to csv files directly in configs.

The next step is to build the image of the container by running : 

```
docker build -t solAI .
```

After that, you can simply run the container by entering : 

```
sudo docker run -it --gpus all --ipc=host --ulimit memlock=-1 --name=solAI -v $(pwd):/code solaria bash
```

To access the container, type :

```
docker container attach solAI
```

Once you are inside the container, you can start training the models. For classification, run : 

```
python classification/train.py
```

and for segmentation, just run :

```
python segmentation/train.py
```

Please note that you also need to update the two config files, one in the classification folder and one in the segmentation folder, with your Comet ML API key before running the training.

You can detach from the container by pressing Ctrl-P followed by Ctrl-Q, and let it continue its training. 

When training is completed, if you want to stop the container, use the command : 

```
docker container stop solAI
```

and you can restart it later on and start a new training if you wish, with : 

```
docker container start solAI
```

We hope you find our project useful and we welcome any feedback you may have !

## Project coverage

[A. Roussi, Solar power shines through after a slow start in Africa, Finantial Times, May 26 2022](https://www.ft.com/content/62c5307a-1877-434e-86b1-7d9fadcdafa2)

[E. Maussion, Sénégal, Burkina, Mali : quand les panneaux solaires arrivent en ville, Jeune Afrique, January 26 2022](https://www.jeuneafrique.com/1302466/economie/senegal-burkina-mali-quand-les-panneaux-solaires-arrivent-en-ville/?utm_source=linkedin.com&utm_medium=social&utm_content=jeune_afrique&utm_campaign=post_articles_linkedin_26_01_2022)

[C. Cosset, Électrification du continent africain: le «deep learning» au service de la recherche, RFI, May 04 2022](https://www.rfi.fr/fr/podcasts/afrique-%C3%A9conomie/20220503-%C3%A9lectrification-du-continent-africain-le-deep-learning-au-service-de-la-recherche)

