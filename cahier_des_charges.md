# Cahier des Charges
Anaelle JAFFRÉ, Thomas Wanchaï MENIER, Pauline PROST–WATERSON, Antonin DE BOUTER

## Description du besoin
Dans un contexte d'armement et de nouvelle technologie, les enjeux liés aux drônes se développent. Les drônes permettent de cibler des points stratégiques à distance qui nécessitent alors d’être défendus. Pour ce faire, leur détection est primordiale. 
Les enjeux de cette démarche sont liés à la survie des dispositifs pour empêcher le drône de remplir ses objectifs :

1. Détecter la présence de drones suffisamment tôt. On peut définir une distance minimale de détection à 10 mètres.

2. Définir la position du drone en temps réel, de façon très précise malgré : 	
    - sa taille assez petite,
    - sa distance du dispositif,
    - son mouvement saccadé et potentiellement imprévisible.

3. Éviter absolument les faux-négatifs, qui laisseraient passer des dangers potentiels.

4. Éviter le plus possible les faux-positifs, incluant le fait de ne pas confondre un drône avec un humain ou un oiseau qui causerait des dommages et victimes collatéraux.

5. Avoir des résultats en temps réel : un temps de latence trop élevé serait trop risqué.

6. La détection doit être efficace peu importe l’environnement incluant : 
    - différentes luminosités possibles,
    - différents paysages possibles (forêt encombrée, champs dégagés, etc.).

### Spécifications et contraintes
Les contraintes techniques imposé par ce projet sont les suivantes : 
- Une unique caméra, bien qu’à terme, plusieurs caméras seront utilisées ;
- Deux types différentes de caméras : 
    - e-CAM_CUOAGK : 2432*2048 px (cercle dans un carré) ;
    - ecam 20 : 
        - Fish Eye : Fov = 185° ;
        - Normale : Fov_h = 85.64° ;
- Type de drône à détecter : Mavic Pro 4.

Pour entraîner nos modèles et tester nos méthodes mises en place, nous utiliserons le dataset mis à notre disposition à l’adresse suivante : https://www.dropbox.com/scl/fo/h5zl7ypbqqof3pmyhe3x3/AHaGiPDtvXlEAc4KFfEhBew?rlkey=p5rw9cmfr0xlfhv3vefxk4ynp&e=1&st=ohem2fvx&dl=0 .

La méthode mise en place doit être déployée sur un système embarqué avec suffisamment de puissance de calcul pour exécuter le traitement de vidéo, la détection de drône et le calcul de position.

## Verrous technologiques

### Détection d’objets
#### Distance de capture et résolution de la caméra
- Si l'objet volant est trop loin de la caméra, il apparaît comme un amas de pixels à l’image et peut ne pas ressembler à un drône. 
- Il peut être confondu avec un oiseau voire un humain.
#### Images par seconde
- Si le taux de rafraîchissement de la détection est trop faible, un drône avec une vitesse élevée peut apparaître déformé à l’image.
- De même, si le nombre de frames par seconde est trop faible, il se peut qu’il n’y ait pas assez de temps alloué pour la détection du drône, calculer puis prédire sa position pour le tir à venir.

### Temps de calcul
Le programme doit fonctionner en temps réel, sur un petit GPU. La complexité des fonctions implémentées doit donc rester faible, idéalement linéaire.

## Description de la solution long terme proposée
Sur le long terme, la solution a pour but d’aboutir à la démarche suivante :
1. Utiliser un **détecteur** pour déterminer la position de l’objet en mouvement dans chaque frame.
2. Y associer un **tracker** (algorithmes de pattern matching) pour obtenir sa position dans l’espace, ainsi que sa vitesse et potentiellement prédire sa trajectoire.
3. Éventuellement, remplacer le **filtre** de Kalman simple implémenté dans le tracker par un filtre plus complexe afin de gérer les mouvements non linéaires. 

### Détecteur
Pour détecter le mouvement, il est possible d’utiliser : 
- Un détecteur codé grâce à des fonctions de détection de mouvements en OpenCV sur Python.

- Un détecteur plus avancé tel que le modèle d’IA YOLO par [Ultralytics](https://www.ultralytics.com/).
Pour ce projet, il a été décidé de tester les deux solutions afin d’évaluer leur pertinence dans le cas donné.

#### OpenCV
Bibliothèque Python utilisée pour le traitement d’image. Permet un traitement temporel via le calcul du flux optique, ainsi que la soustraction d’arrière plan.

#### YOLO
La seconde option est d’utiliser un algorithme de détection tel que Yolo pour la détection d’objets en mouvement, avec les **trois canaux usuels** ainsi qu’un **quatrième canal** correspondant au **mouvement**. Ce dernier canal servira à reconnaître le drône grâce aux spécificités de son déplacement, qui peuvent le différencier d’un oiseau ou d’un humain par exemple.

Il sera important de fine-tuner le modèle avec plus de données représentatives pour améliorer les performances et bien incorporer le quatrième canal dans le modèle ainsi que de prendre en compte le système final sur lequel sera monté la caméra. Cela permettra, par exemple, de prendre en compte le mouvement de la caméra, s'il y en a. 

### Tracker
Le détecteur permet de repérer un objet en mouvement. Si l’on souhaite obtenir sa **trajectoire**, sa **vitesse** et sa **position** en continue, il est possible d’y combiner un système de tracking.

Pour le tracking, on peut utiliser des modèles légers tels que OC-SORT ou BYTETrack. Ce sont des améliorations du tracker SORT (Simple Online and Real-time Tracking) : en plus de donner les sorties attendues d’un tracker, ils peuvent fonctionner sur des mouvements potentiellement non linéaires, ou des objets qui subissent une occlusion. 

#### OC-SORT
OC-SORT (Observation-Centric SORT) propose trois solutions pour surpasser les limitations de SORT :
- Observation Centric Re-Update (ORU) ;
- Observation Centric Momentum (OCM) ; 
- Observation Centric Recovery (OCR).

Ces améliorations permettent de corriger la prédiction linéaire avec des ajustements centrés sur l’objet. L’ORU et l’OCM sont des solutions efficaces pour contrer l’occlusion et le mouvement non linéaire. L’OCR est une heuristique, qui vise à empêcher la perte de pistes grâce à une association secondaire. Grâce à ces trois aspects, l’OC-SORT est idéal dans le cas de changements brusques de vitesse. Il est donc pertinent de l’utiliser pour le suivi d’un drône.

#### BYTETrack
BYTETrack est un tracker orienté pour du multi-objet. Son principe de fonctionnement est différent de SORT dans la mesure où il ne rejette pas les détection à faible confiance qui ne correspondent pas à l'arrière-plan. Elles sont simplement associées à un score faible, et passent dans une seconde étape d’association avec des pistes existantes. Cela permet de toujours suivre un objet donné, même en cas d’occlusion ou de flou.

#### Choix du tracker
Les deux trackers peuvent être pertinents pour le suivi d’un drône. OC-SORT est adapté au suivi d’un objet donc la vitesse peut changer brutalement, et BYTETracker est idéal pour garder le suivi de l’objet en mouvement, même en cas de scène peu lisible. Il faudrait évaluer lequel donne de meilleurs résultats dans le cadre du projet, selon les différentes situations données.

### Filtre de Kalman
Si un tracker issu du modèle SORT est choisi pour le projet, il faut spécifier, au sein du programme, le filtre choisi pour le **débruitage** et la **prédiction**. Il existe différents types de filtres Kalman pour ce faire. Initialement, on pourrait implémenter un modèle simple, pour minimiser le temps de calcul.

Cependant, ce choix peut avoir des limites : la trajectoire d’un drône est souvent **non linéaire**. Il peut enchaîner des phases stables avec des mouvements rapides, des changements de vitesse fréquents. Dans le cas d’un choix orienté OC-SORT, cela ne devrait pas être un problème, mais il peut s’avérer utile pour une implémentation avec BYTETrack.

Pour cela, on peut remplacer le filtre de Kalman simple par un modèle non-linéaire, comme l’UKF (Unscented Kalman Filter) ou l’IMM (Interacting Multiple Model).

#### UKF
Le filtre UKF utilise l’Unscented Transform, qui approxime la transformation non linéaire en l'appliquant à un ensemble de points d'échantillonnage. Il est capable de gérer les non linéarités douces, sur un mouvement continu.

#### IMM
L’IMM est un modèle qui combine plusieurs filtres de Kalman simples pour différents objectifs. Il possède une complexité proportionnelle au nombre de filtres utilisés. On peut utiliser trois critères pour définir les filtres choisis : vitesse, accélération et rotation. Dans ce cas, le modèle comprendra un ensemble de trois filtres, chacun dédié à l’un de ces critères.

#### Choix du filtre
Dans le cas d’un tracking de drône, il est préférable d’opter pour un modèle de filtre comme l’IMM, puisqu’il peut prendre en compte les changements brusques. Cependant, sa complexité est légèrement plus élevée que celle d’un UKF. Le temps de calcul est à minimiser, et peut-être que l’implémentation d’un filtre non linéaire n’est pas assez rentable au vu du temps sacrifié. Ainsi, pour choisir un filtre adapté, il serait préférable de faire des tests avec le matériel à disposition.

## Description du prototype qui sera réalisé
Les contraintes temporelles actuelles du projet impliquent une restriction des choix techniques. Ainsi, pour donner une première réponse à la problématique sur le court terme, des décisions figées ont été prises au préalable.

### Détecteur
Afin de détecter le drône en mouvement, il a été décidé de développer en parallèle un modèle personnalisé de **YOLO** ainsi qu’un algorithme de détection par **OpenCV**.

L’objectif est de tester si OpenCV est préférable à Yolo pour la détection de mouvement simple. Le critère de jugement est le rapport de la performance (taux de détection réussie) sur le coût computationnel. Il est souhaitable d’avoir un taux de détection réussie élevée et un temps de calcul faible, ce rapport est donc à maximiser.

### Tracking et filtre
Pour la récupération des informations de position et de mouvement, deux options se présentent suite aux solutions évaluées.

BYTETrack est idéal pour détecter les objets en zone de flou ou qui subissent une occlusion, mais il est préférable de l’accompagner d’un IMM pour plus de performances, ce qui augmente le coût computationnel. OC-SORT est privilégiable pour la détection d’objets à vitesse changeante et à faible coût, mais peut avoir des résultats moins satisfaisants que la première option.

Dans une optique d’optimisation de la détection, le choix sur le court terme s’oriente donc sur une solution **BYTETrack** accompagnée d’un **IMM** implémenté en filtrage.

### Notes fonctionnelles
Les coordonnées dans l’espace 3D physique seront données seulement pour une caméra fixe, et non en mouvement.






