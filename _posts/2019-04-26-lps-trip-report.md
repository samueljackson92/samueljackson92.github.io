---
layout: post
title: LPS Trip Report
tags: conference, earth observation, machine learning
---

*This is a collection of my notes from the Living Planet Symposium 2019 which was
held in the beautiful city of Milan, Italy. The conference only comes around every
three years, so I was lucky to start my current project with CEDA in time to visit.*

## Monday

### Morning: Opening Session

The moring welcome session opened after registration at 9am in the Main auditorium of the Milano Congressi conference center. I missed the opening welcome video due to the center being so large that I took me 5 minutes to walk from registration of the main autitorium. 

![IMG_20190513_091530](/assets/images/IMG_20190513_091530.jpg)

<p class="center">*View from the back of the main auditorium*</p>

The first sessions were a combination of addresses by a series of directors and official from various earth observation programmes and initatives. Most of the talks in these sessions highlighted at a very high level the past successes and future challeneges for the European Earth observation programmes. Below are some (low quality) photos of some of the more important slides from the session.

![IMG_20190513_092543](/assets/images/IMG_20190513_092543.jpg)



![IMG_20190513_093050](/assets/images/IMG_20190513_093050.jpg)

<p class="center">*Slides from Johann-Dietrich Worner's (Director General, ESA) talk*</p>

Several speakers pointed out the importance AI and machine learning in the furture of earth observation, both in the sense of classic applciation to image preprocessing (classifications, segementation, etc.) but also  in terms of new directions such as collision avoidance and automated de-orbiting to prevent the creation of disaterous space debris. The importance of this was highlighted by an example of a solar panel failure in Sentinel-1a.

One of the most interesting slides from the session was from Josef Aschbacher's (DIrector of EO Programmes, ESA) slides on the "Pushing AI beyond the final frontier". ESA are planning to launch a [satellite](https://spacenews.com/esa-preps-earth-observation-satellite-with-onboard-ai-processor/) with a [Intel Myriad processor](https://www.movidius.com/myriadx) VPU on board, bringing the advantages of edge computing to the earth observation domain.

![Screenshot 2019-05-13 at 12.23.52](/assets/images/Screenshot 2019-05-13 at 12.23.52.png)

Other themes present in the keynotes were pretty much what you'd expect from a EO conference:

- Better engagement & communication with the non-science community
- The political and policy implications enabled by EO.
- The increasing relevance of EO as the climate crisis intensifies.
- Maintaining high quality of datasets as volume increases
  - Also the importance of reprocessing: can we really throw things away?
  - Example of the O-zone hole only being discovered only after multiple satellites confirmed the same systematic discrepancy.
- The continuing importance of open data for engaging the wider community.
- Several slides underscoring the fact that EO puts more into the European economy that it takes out.

The last talk by young activist Jakob Blasel was very interesting and moving; slamming the lack of political action on the climate change from both EU and global leaders. He underscored the importance of EO in producing facts upon which to build goverment and international policy. Unsurprisingly this talk was met with much applause.

### Afternoon

#### The GEO Knowledge Hub: Supporting the Earth Observation Community to Use Big Data Analytics 

 *Gilberto Camara, Geo - Group On Earth Observations, Geneva, Switzerland*

An impassioned talk outlining the need for the EO community to avoid just pumping out papers and to engage with the world at large. This talk spoke to me as an ex-software engineer. Gilterto made clear the distrinction between research and decision making and defined the "Valley of Death" inbetween. Outlined the importance of trying to incentivise bridging this gap.

He pointed out the importance that scientific results should be more than just papers. There's a spectrum of reproducibility from just a publication through to fully repdorucable code + paper + data + runnability. There should be an empshais of making code easy to run and reuse.

Outlined the need to move away from giving the data to the end user. We need to provide platforms and services that enable to community to access and work with the data without transfers. This correlates perfectly with a discussion I had earlier in the day about how hard it is to share data in africa due to network infrastructure availability.

![IMG_20190513_150105](/assets/images/IMG_20190513_150105.jpg)

![IMG_20190513_150338](/assets/images/IMG_20190513_150338.jpg)

![IMG_20190513_150207](/assets/images/IMG_20190513_150207-7761645.jpg)

![IMG_20190513_150207](/assets/images/IMG_20190513_150748.jpg)

#### Revealing the Network - Road Extraction from Satellite Imagery for Security Applications Using a Deep Learning Approach

*Kristin Fleischer, IABG mbH, Ottobrunn, Germany*

An interesting talk on using machine learning to the problem of road extraction in very high resolution (VHR) satellite images. They used an ensemble of U-Nets trained on their data to first produce a segementation mask for the raw satellite images. From their the applied a vectorisation step to their data to convert from the raw U-Net predictions to a vector based representation that also cleans up unwanted artifacts such as imperfections at road junctions. The application machine learning has allowed her company to automate a process that was previously largely manual and labour intensive. Their requirements for successfully application were:

- High spatial accuracy
- Very timely results, near real time.
- High resolution road networks

One interesting point she made was that getting a representative distribution of images was not trivial. Not only is it labour intensive to obtain, but often their clients are interested in "non touristic" domains where their is a sparsity of data. Not every place of interest is well covered by remote sensing systems.

Another interesting experiment which they performed was to try transfer learning from the VHR images to lower resolution satellite images from Sentinel-2. They found that they achieved mixed performance with the network performing well in rural areas, but it failing in urban areas, probably due to the loss of local information with the lower resolution.

![IMG_20190513_165021](/assets/images/IMG_20190513_165021-7762642.jpg)

![IMG_20190513_165148](/assets/images/IMG_20190513_165148-7762774.jpg)

![IMG_20190513_165517](/assets/images/IMG_20190513_165517-7762794.jpg)

![IMG_20190513_165741](/assets/images/IMG_20190513_165741.jpg)

![IMG_20190513_165935](/assets/images/IMG_20190513_165935.jpg)



## Tuesday

#### Deep Learning to Retrack Inland Waters Waveforms From Radar Altimetry Satellites Sentinel-3 and Jason-3

*Denis Blumstein, LEGOS, Toulouse, France; CNES, Toulouse, France*

This project demonstrated the use of combining radar altimetry readings together to form a 2D image (Radargram). Each column in the image equates to a radar echo. The intensity of each pixel is the power of the echo response. Radargrams over inland water can exhibit a lot of complexity. 

They formulate the problem as a regression problem, attempting to find a mapping $f(R) \rightarrow height$ in the presence of a lot of noise. They used a model based on Res-Net-50 but with some added dropout layers and using a single output ReLU head. Importantly, all of their training data was simulated so they could create ~96,000 samples. In their results they found the could obtain 90% accuracy on the simulated data.

They compared against a limited number of real "ground truth" data and showed they could get a decent regression model from it. A final slide that I did not manage to get a photo of showed that they did some visualisation of the weights are found at least some correlation with what they would expect to be good features for this kind of data.



![IMG_20190514_091120](/assets/images/IMG_20190514_091120.jpg)

![IMG_20190514_091321](/assets/images/IMG_20190514_091321.jpg)

![IMG_20190514_091557](/assets/images/IMG_20190514_092007.jpg)

![IMG_20190514_092252](/assets/images/IMG_20190514_092252.jpg)

#### The MORINGA Processing Chain: Automatic Object-based Land Cover Classification of Tropical Agrosystems using Multi-Sensor Satellite Imagery

*Raffaele Gaetano*, Cirad - Umr Tetis, Montpellier, France

A talk presenting the MORINGA processing framework from argricultural classification. A complex framework combining both VHR data and time series data from the Sentinel-2 and Landsat-8 missions. This is a challenging problem with many classes to consider, which exhibits considerable diversity between regions. Size and scale of classification regions varies by farming practices between regions. Generally you have limited or noisy reference data (as I would assume, the "tourist problem" as well.). They also noted that their problems are heavily affected by cloudy conditions over the regions of interest, so time series data is critical.

Results seemed quite mixed between with good F1 scores in some crops classes, but quite bad in others. They also showed a lot of variability between different crop scenes, with the classifier apparently performing better in more homogeneous regions.



![66a812a84e9c45fca469ffed2917775a](/assets/images/66a812a84e9c45fca469ffed2917775a-7822815.png)

![66a812a84e9c45fca469ffed2917775a](/assets/images/IMG_20190514_093851.jpg)

![IMG_20190514_094006](/assets/images/IMG_20190514_094006-7822860.jpg)

![IMG_20190514_094338](/assets/images/IMG_20190514_094338.jpg)

![IMG_20190514_094800](/assets/images/IMG_20190514_094800-7823188.jpg)

![IMG_20190514_094526](/assets/images/IMG_20190514_094526.jpg)

![IMG_20190514_095013](/assets/images/IMG_20190514_095013.jpg)

![IMG_20190514_095244](/assets/images/IMG_20190514_095244.jpg)



#### Deep learning techniques for the quality enhancement of hyperspectral images

*Giorgio Licciardi, Hypatia Research Consortium, Rome, Italy; GIPSA-Lab, Grenoble, France*

Giorgio presented a denoising approach for hyperspectral imagrey base on using an auto encoder. The name of the apporach is [NLPCA]([http://www.nlpca.org/](http://www.nlpca.org/)). The found that the could mitigate the noise effects of Gaussian noise, banding failures, "smile effect"  (curved spatial distortion from optics), and atmospheric contributions.

This talk appeared to be a pretty straight forward application of an existing ML algorithm to EO data. The author didn't mention any particular novelties that differed from the reference implementation other than adapting the input/output size to match the hyperspectral data.

![IMG_20190514_120034](/assets/images/IMG_20190514_120034.jpg)

![IMG_20190514_120125](/assets/images/IMG_20190514_120125.jpg)

![IMG_20190514_120510](/assets/images/IMG_20190514_120510.jpg)

![IMG_20190514_120713](/assets/images/IMG_20190514_120713.jpg)

![IMG_20190514_120937](/assets/images/IMG_20190514_120937.jpg)

![IMG_20190514_121154](/assets/images/IMG_20190514_121154.jpg)



### Afternoon

#### Spatially-aware intercalibration between SPOT-VGT and PROBA-V

*Julien Radoux, Uclouvain, Louvain-la-Neuve, Belgium*

Julien's talk wasn't hugely relevant to my area of interest but he did make a excellent point regarding combining data at different resolutions from different platforms. In his talke he showed that combining data from PROBA and SPOT. When they resampled the data from PROBA from 300m to a 1km grid to match SPOT they noticed that they were getting a blurring in resolution from SPOT. PROBA are inherently sharper because of a difference in point spread function as perfectly illustrated in the diagram from his slides.



![IMG_20190514_140117](/assets/images/IMG_20190514_140117.jpg)

![IMG_20190514_140215](/assets/images/IMG_20190514_140215.jpg)

![IMG_20190514_140608](/assets/images/IMG_20190514_140608.jpg)

![IMG_20190514_140626](/assets/images/IMG_20190514_140626.jpg)



#### Fractional Land Cover Mapping of Africa Using Machine Learning Techniques on Proba-V Image Time Series

*Dainius Masiliunas, Wageningen University & Research, Wageningen, Netherlands*

The idea behind this project is to treat pixels in satellite images not as a discrete class, but as potentially a combination of multiple classes in a ratio that sums to unity. The speaker tried a wide variety of different models, variables and metrics. Interestingly the dataset they used were a training and test set both gathered over the African continent, but the speaker noted that the validation set was a completely different dataset from the one used for training. This means that this is an excellent of generalisation because even though the share similar labels, they might differ slightly in how the specifically define a class (i.e. what makes a shrub a shrub?).



![IMG_20190514_140904](/assets/images/IMG_20190514_140904.jpg)

![IMG_20190514_141114](/assets/images/IMG_20190514_141114.jpg)

![IMG_20190514_141221](/assets/images/IMG_20190514_141221.jpg)

![IMG_20190514_141301](/assets/images/IMG_20190514_141301.jpg)

![IMG_20190514_141703](/assets/images/IMG_20190514_141703.jpg)


![IMG_20190514_141808](/assets/images/IMG_20190514_141808.jpg)

![IMG_20190514_141937](/assets/images/IMG_20190514_141937.jpg)



## Wednesday

### Morning

#### Large-scale Automatic Agricultural Parcel Delineation Using Sentinel-2 Imagery and a Convolutional Neural Network

*Kristof Van Tricht, VITO Remote Sensing, Mol, Belgium*

A study of applying a CNN over agricultural imagrey captured over Belgium. Focuses on a single tile from Sentinel-2 data. For input they use all available stacks for a particular tile with NDVI comuted from the channels. They note in the talk abstract that 

>  "*idea behind the method is that individual agricultural parcels behave differently along the growing season, and hence their spectral signature will have a field-specific behavior*"

The model architecture used was a U-Net.

![IMG_20190515_114652](/assets/images/IMG_20190515_114652.jpg)



![IMG_20190515_114917](/assets/images/IMG_20190515_114917.jpg)



![IMG_20190515_115413](/assets/images/IMG_20190515_115413.jpg)



### ![IMG_20190515_115503](/assets/images/IMG_20190515_115503.jpg)

### Afternoon

#### Innovative Deep Learning Solutions for the Improvement of Agriculture Monitoring Capabilities in Europe

*Veronique Defonte, CS Systèmes d’Information, Toulouse, France*

They present a hybrid architecture of both a CNN and an RNN for crop monitoring. Their main motivation for combining the two architectures was that they both fail in different ways and that combining their results leads to better performance. They note:

- A CNN oversmooths crop parcels leading to boundaries that are too smooth
- A RNN undersmooths crop parcels leading to ragged boundaries and isolated pixels

The CNN backbone used was ResNet3D and they use a combination of RGB, NIR, and elevation data as well as band combinations. For the RNN they use the same features as for the CNN but use an LSTM model for temporal, pixel wise classification.

They combined the two by using a 2D CNN classifier on induvidual patches, then using a RNN model to classifier they same patches but for time series data. This combination gave the best results, but they noted that they had issues with interpolation with clouds. Further work will look into combining their model with data from Sentinel-1 to combat this.

![IMG_20190515_155710](/assets/images/IMG_20190515_155710.jpg)![IMG_20190515_160000](/assets/images/IMG_20190515_160000.jpg)

## ![IMG_20190515_160028](/assets/images/IMG_20190515_160028.jpg)

![IMG_20190515_160154](/assets/images/IMG_20190515_160154.jpg)



![IMG_20190515_160300](/assets/images/IMG_20190515_160300.jpg)



![IMG_20190515_160611](/assets/images/IMG_20190515_160611.jpg)



## Thursday

*Thursday morning had the main machine learning/deep learning stream at LPS, with the most relevant talks in my area. My phone battery died mid way through the day so notes on the afternoon sessions are more sketchy*

### Morning

#### Memory effects of climate and vegetation affecting net ecosystem CO2 fluxes in global forests

*Simon Besnard, Max Planck Institute For Biogeochemistry, Jena, Germany; Laboratory Of Geo-information And Remote Sensing, Wageningen University, Wageningen, Netherlands*

A talk showing the use of a RNN to capture the temporal evolution of $CO_2$ flux between atmosphere and forests. Specifically the project was attempting to predict Net Ecosystem Exchange (NEE) which is defined as the ecosystem respiration less the effects of photosythesis. As one of the slides shows this some function $f$ from $f(vegetation state, climate) = NEE$. The temporal data used was from both Landsat and MODIS imagery.

![IMG_20190516_085433](/assets/images/IMG_20190516_085433.jpg)



![IMG_20190516_085553](/assets/images/IMG_20190516_085553.jpg)



![IMG_20190516_090156](/assets/images/IMG_20190516_090156.jpg)

#### Lithological classification using multi-sensor data and Convolutional Neural Networks

*Melanie Brandmeier, Esri Deutschland Gmbh, Kranzberg, Germany*

Melanie gave a talk on using a U-Net classification model for lithological mapping of a mining region in northern Australia. Essentially the problem is similar to crop or region classification. They used a combination of Sentinel-2A and ASTER data along with some coarse geophysical data to perform region classification. Two things of interest to note from this talk, 1) their U-Net was not particularly deep, 2) an ablation experiment with and without the geophysical data *decreased* the accuracy of the network, but they were concatenating the data as an additional channel at the front of the network. 

This made wonder if the network was "blurring out" the effect of the additional info during the compression stage and whether concatenating the data onto the end of the network would yield better results. Alternatively, perhaps there are more integlligent ways to incorporate it as a prior.



![IMG_20190516_090645](/assets/images/IMG_20190516_090645.jpg)



![IMG_20190516_090914](/assets/images/IMG_20190516_090914.jpg)



![IMG_20190516_091217](/assets/images/IMG_20190516_091217.jpg)

#### Deep Learning for Damage Estimation Using High-Resolution UAV Imagery

*Ferda Ofli, Qatar Computing Research Institute, Doha, Qatar*

Talk on using images caputed from UAVs over distaster zones for the assessment of response needs. The real challenge of this project is the high variation in the number of types of disaster, as well as the diverse envrionments that classifier must operate in. Additionally there is high variance in the angle and viewpoint of the images. The main case study presented in the talk was to identify damaged buildings after a cyclone in Vanuatu. Not only do they wish to indentify building as damanged, but also to classify the level of damage into a discrete class (low/med/high damage). 

They compared the use of two popular architectures: Faste R-CNN and YOLO (v2). Results from the experiments were fairly poor compared to the same architectures applied to standard benchmarks, which is probably to be expected given the scale of the problem. They noted that they were limited by the accuracy of the ground truth where their were many unannotated or poorly annotated images or label ambiguity from multiple experts. They noted in their take home messages that dealing with noisy ground truths is just part of the game with EO data.



![IMG_20190516_092654](/assets/images/IMG_20190516_092654.jpg)



![IMG_20190516_093010](/assets/images/IMG_20190516_093010.jpg)

#### A Recurrent Neural Network (RNN) Based Approach for Reliable Classification of Land Usage from Satellite Imagery

*Prateek Purwar, CSEM SA, Alpnach Dorf, Switzerland*

A project to apply RNN apporaches to model the spectral evolution of pixels in a scene with the goal of classifying land usage. They first clean their images using an RNN auto encoder to smooth out the signal different land usage types. A classifier is then trained on the smoothed output to give a probability of the particular class. One classifier is trained per class and the resulting class if a majority vote. (This made me curious about whether they tried just using categorical cross entropy?).

![IMG_20190516_094301](/assets/images/IMG_20190516_094301.jpg)



![IMG_20190516_094410](/assets/images/IMG_20190516_094410.jpg)



![IMG_20190516_094441](/assets/images/IMG_20190516_094441.jpg)



![IMG_20190516_094819](/assets/images/IMG_20190516_094819.jpg)

### Deep Learning for Global Local Climate Zones Classification

*Xiaoxiang Zhu, German Aerospace Center (DLR) , Wessling, Germany; Technical Universizy Of Munich (TUM), Munich, Germany*

They presented a new dataset of local climate zones derived from 400,673 images from Sentinel-1, 2 and TanDEM-X  with ground truth generated by human experts. On top of this dataset they used a CNN to extract local features and a RNN to perform temporal modelling. The CNN used a ResNet backbone and the RNN model was based on LSTMs. Notably, they gave significant thought to how they split their training & test sets. They used three different methods:

- Random splitting: just sample!
- Block splitting: Splitting based on geographic blocks
- Culture splitting: splitting based on regions with distinct types of urban environments. E.g. US cities look different from Asian cities!

With this dataset they produced the worlds first global, urban local climate zone classification.



![IMG_20190516_100545](/assets/images/IMG_20190516_100545.jpg)

#### Dealing With Weak Labelling and Extremely Unbalanced Data: Training Deep Convolutional Neural Networks to Detect Multi-pixel Ships Using Single-pixel Labels in SAR Images

*Stian Normann Anfinsen, UiT Arctic University of Norway, Tromso, Norway*

A project aimed at finding ships at sea in Sentinel-1 SAR images. They noted that they only used weak labels for their ships. Each ship was manually identified by a human expert using a single pixel. They then used a region growing apporach to capture the entirety of the ship. For their architecture they try two models. One is a CNN encoder followed by a 8x8 binary classification. The second architecture is a U-Net.



![IMG_20190516_104828](/assets/images/IMG_20190516_104828.jpg)



![IMG_20190516_105358](/assets/images/IMG_20190516_105358.jpg)

#### CONTINUAL LEARNING FOR DENSE LABELING OF REMOTE SENSING IMAGES

*Onur TASAR, INRIA, Sophia Antipolis, France*

Interesting talk by Omar on the challenges face by the EO community when attempting to apply deep learning techniques in the wild. The group is investigating continual learning, without having to store and retrain networks using a enormous amount of data. They use a fusion style approach that combines a couple of U-Nets (see image below). They use a custom loss function to try an mitigate the effects of catastropic forgetting when the netowkr is shown new information, and weight this with the classification loss. In this way they show you can train a network on one problem (e.g. land cover) then retrain it on another (e.g. water classification) and end up with a classifier that does both, without training two classifiers from scratch.

They also note that not all patches used for training are of equal importance, they show a method of weighting image patches by importance to store only 30% of the total amount of data for retraining. Some limitations they note are that the current method only works when all training data have similar spectral distributions.

![IMG_20190516_110134](/assets/images/IMG_20190516_110134.jpg)



![IMG_20190516_110201](/assets/images/IMG_20190516_110201.jpg)

![IMG_20190516_110355](/assets/images/IMG_20190516_110355.jpg)



![IMG_20190516_110553](/assets/images/IMG_20190516_110553.jpg)

#### Rotation equivariant CNNs for the semantic labeling of remote sensing imagery

*Devis Tuia, Wageningen University, Wagenignen, the Netherlands*

A really interesting talk presenting an architecture I had not seen before. They present their work on land cover classification using [RotEqNet](https://arxiv.org/pdf/1612.09346.pdf) which handles the case where equivariance is required by learning a vector field with angle and magnitude. 

This topic is really interesting because it hits home on a fundermental point about the type of datasets present in EO images: there is some redundancy in the fact that images + labels can be rotated an mean the same thing. The presenter noted that the field should think more about the type of geometric transformation that apply to their data. 

I definitely think more reading on this topic would be worth following up:

- https://arxiv.org/pdf/1612.09346.pdf

- https://www.sciencedirect.com/science/article/abs/pii/S0924271618300261



![IMG_20190516_111753](/assets/images/IMG_20190516_111753.jpg)



![IMG_20190516_111947](/assets/images/IMG_20190516_111947-8969476.jpg)

#### Counting the Uncountable: Deep Demantic Density Estimation From Space

*Andres Camilo Rodriguez, ETH Zurich, Zurich, Switzerland*

Another really interesting talk. What do you do when you want to count objects in images where you can't independantly resolve induvidual objects? This is a nice approach to really try and squeeze the most out of the data available to you. They presented a couple of examples of the density estimation approach and contrasted it with some ground truths. They show examples of counting cars in a parking lot and counting olive trees in farmland. 

They use a ResNet backbone which ends with two seperate paths: one for semantic segementation of regions and one for density estimation of induvidual pixels. The loss function is the sum of the loss of the two paths. They showed their results can count to within 5% accuracy of the ground truth.



![IMG_20190516_113342](/assets/images/IMG_20190516_113342.jpg)



![IMG_20190516_113540](/assets/images/IMG_20190516_113540.jpg)



![IMG_20190516_113715](/assets/images/IMG_20190516_113715.jpg)

#### Deep Features for Change Detection in Multitemporal VHR SAR Images

*Sudipan Saha, Fondazione Bruno Kessler, Trento, Italy*

This was the only talk of the week which featured a GAN. In their work the authors look at the problem of change detection in multiple VHR SAR images. As many speakers noted, it is difficult to obtain a large amount of labelled data in the EO domain. However, there is plenty of unlabelled image pairs.

The architecture they use is based on the [cycle consistent GAN]([http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)) for feature extraction, followed by feature comparision between the image pairs. For further details on the work, follow up on the [DOI]([10.1109/IGARSS.2018.8519440](https://doi.org/10.1109/IGARSS.2018.8519440)).



![IMG_20190516_115001](/assets/images/IMG_20190516_115001.jpg)



![IMG_20190516_115117](/assets/images/IMG_20190516_115117.jpg)



![IMG_20190516_115608](/assets/images/IMG_20190516_115608.jpg)

![IMG_20190516_115852](/assets/images/IMG_20190516_115852.jpg)



#### Data Augmentation Techniques for Deep Learning Based Satellite Image Super-resolution

*Moataz Ahmed, School of Environmental and Geographical Sciences, University of Nottingham Malaysia, Semenyih, Malaysia*

A simple but nice exploration of applying a super resoltuion encoder to Landsat and Sentinel-2 data. They applied different image augmentations common in machine vision to examine the effect of this on performance of super resolution reconstruction. The results showed that the augmentation (unsurprisingly) improved performance.

#### Automatic Wake Detection on SAR Images by Deep Convolutional Neural Networks

*Corrado Avolio, e-geos, Roma, Italy*

While the application of machine learning to ship detection had alreayd been covered earlier in the day, Corrado spoke on an apporach based on detecting ships from their wake patterns for SAR images. This is often complicated by the noise in the images, artifacts from the sea etc. A major challenge in this domain was dealing with the varied and non linear shape of wake patterns.

#### Efficient simulations of microwave radiometry observations of the atmosphere at very fine spatial resolution: an application of the combination of a neural network with a xarray/dask/parquet parallel computing environment for EO applications.

*Bruno Picard, Fluctus, Rabastens, France*

A varied presentation which spent a fair amount of time prasing Dask and xarray. One interesting take away was the use of trying to replace radiative transfer models with machine learning as this could considerably cut down computation time for NWP.

#### Machine Learning for Identifying and Monitoring Mongolian Infrastructure to Support Sustainable Pastureland Management

*Anneley Hadland, Deimos Space, Harwell Space Cluster, United Kingdom*

A very interesting talk from Anneley talking about the application of machine learning algorithms to image data collected from Mongolia. She presented a couple of different cases applications including:

- Migration path indentification
- Yurt detection in urban scenes
- Campsite detection (both current and historical) from the distinctive pattern left on the land.

She noted that challenges faced such as (as always) a lack of ground truth, but also made some important points about other problems faced by EO. For example, in the migration path identification problem, it is not clear when a trail becomes a path which becomes a road? How do we smooth detections to "cartographically correct" versions?

### Conclusions & Final Thoughts

My first LPS experiance was excellent. I learnt a lot about the challenges faced by the community and what other people beyond my own little sphere are paying attention to. It also gave me some ideas for how I can further my own work. To summarise the trip I'd like to pull out some of my general thoughts from the conference:

##### Classification & Segmentation are the most common problems

Most of the work I saw related to machine learning can either in the form of a classification problem or a segmentation problem. A few talks discussed instance segmentation problems, and fewer talked about regression problems. Most of the data were from raw satellite imagrey or from UAVs. The most common application area seemed to be crop classification/segmentation.

By far the most common architecture was the classic U-Net, while some variants were mentioned (e.g. Cycle GANs) almost all architectures involved some kind of encoder-decoder based on U-Net. A couple of architectures in the poster session were using dialted convolutions, but this was limited. The other recurring architecture was LSTMs. LSTMs were either mentioned on their own or combined in some way with a CNN conmponent. For pure classification problems, usually the architecture of choice was Faster-R-CNN

##### EO Specific Data Problems

Earth observation has a number of specific problems that make it unique from "general" machine learning. This is an important point because academics in this area are looking at the state of the art on benchmarks which are *not* EO data and trying to transfer to their specific problems.

A couple of points were hammered home throught the conference that are siginifcant for EO data. 

1. EO data problems are often rotation equivariant which can cut down on the required complexity, but can pose problems for creating an architecture that will reflect and respect this.
2. Segmentation problems can often have rough, fine grained edges and SOTA segmentation approaches might oversmooth.
3. EO data from satellites is highly varied with location, season, time, and environmental conditions.
4. EO has a lot of data, most of it is quite large, and almost all of it is unlabelled. This is probably the most important challenge.
5. EO data is pretty accessible and well catalogued compared to other fields. This is likely due to the size of the data (which some areas have not yet caught up with) and the relatively few data generation facilities (i.e. satellites), but ground truth datasets for benchmarking are still lacking.
