# ChallengeAI w/ HeadmindPartners
Ce dépôt contient notre travail pour le ChallengeIA, faisant partie de notre troisème année à CentraleSupélec en mention IA


## Consignes
#### Données :

Le dataset Dior est composé de deux dossiers et d'un fichier csv : 
`
data/
│
├── DAM/
│   ├── 01BB01A2102X4847.jpeg
│   ├── ...
│
├── test_image_headmind/
│   ├── image-20210928-102713-12d2869d.jpeg
│   ├── ...
│
└── product_list.csv
`

- Le dossier "DAM" contient tous les jpeg de références de chaque article (2 766 articles). Le nom de chaque jpeg correspond à son MMC référencé dans le csv. Chaque image est de taille 256x256.

- Le deuxième dossier "test_image_headmind" contient les images test (80 images test). Tous les articles compris dans ces images sont référencés dans DAM et le fichier csv. La taille de ces images est variable. Les images ne sont pas annotées. Le nom du fichier correspond à la nomenclature donnée par l'appareil photo.

- Le fichier csv "product_list" comprend le code MMC unique à chaque article ainsi que le Product_BusinessUnitDesc précisant la classe de l'article (Bags, Shoes, etc)
 
 
#### Objectif :

Le but du projet est de retrouver la référence d'un article à partir d'une photo de ce dernier. Il faut donc utiliser les caractéristiques visuelles des objets afin de retrouver l'article.
 
*Exemple* : Par exemple à partir de l'image `./test_image_headmind/IMG_6880.jpg` , le modèle doit renvoyer l'image `./DAM/BOBYR1UXR42FR.jpeg`.

## Méthode

 
