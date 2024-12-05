# A prospective approach for human-to-human interaction recognition from Wi-Fi channel data using attention bidirectional gated recurrent neural network with GUI application implementation
### DOI: [https://doi.org/10.1007/s11042-023-17487-z](https://doi.org/10.1007/s11042-023-17487-z)
### Citation:
```
@article{Khan_Shams_Raihan_2024, title={A prospective approach for human-to-human interaction recognition from Wi-Fi Channel data using attention bidirectional gated recurrent neural network with GUI application implementation}, volume={83}, DOI={10.1007/s11042-023-17487-z}, number={22}, journal={Multimedia Tools and Applications}, author={Khan, Md Mohi and Shams, Abdullah Bin and Raihan, Mohsin Sarker}, year={2024}, month={Jan}, pages={62379–62422}}
```
### Abstract: 
Human Activity Recognition (HAR) research has gained significant momentum due to recent technological advancements, artificial intelligence algorithms, the need for smart cities, and socioeconomic transformation. However, existing computer vision and sensor-based HAR solutions have limitations such as privacy issues, memory and power consumption, and discomfort in wearing sensors for which researchers are observing a paradigm shift in HAR research. In response, WiFi-based HAR is gaining popularity due to the availability of more coarse-grained Channel State Information. However, existing WiFi-based HAR approaches are limited to classifying independent and non-concurrent human activities performed within equal time duration. Recent research commonly utilizes a Single Input Multiple Output communication link with a WiFi signal of 5 GHz channel frequency, using two WiFi routers or two Intel 5300 NICs as transmitter-receiver. Our study, on the other hand, utilizes a Multiple Input Multiple Output radio link between a WiFi router and an Intel 5300 NIC, with the time-series Wi-Fi channel state information based on 2.4 GHz channel frequency for mutual human-to-human concurrent interaction recognition. The proposed Self-Attention guided Bidirectional Gated Recurrent Neural Network (Attention-BiGRU) deep learning model can classify 13 mutual interactions with a maximum benchmark accuracy of 94% for a single subject-pair. This has been expanded for ten subject pairs, which secured a benchmark accuracy of 88% with improved classification around the interaction-transition region. An executable graphical user interface (GUI) software has also been developed in this study using the PyQt5 python module to classify, save, and display the overall mutual concurrent human interactions performed within a given time duration. Finally, this article concludes with a discussion of the possible solutions to the observed limitations and identifies areas for further research. Such a Wi-Fi channel perturbation pattern analysis is believed to be an efficient, economical, and privacy-friendly approach to be potentially utilized in mutual human interaction recognition for indoor activity monitoring, surveillance system, smart health monitoring systems, and independent assisted living. 

----------------------------------------------------------------------------------------------------

# Software Demo: [![Watch the video](https://img.youtube.com/vi/yUKANUwgi4s/maxresdefault.jpg)](https://youtu.be/yUKANUwgi4s)

----------------------------------------------------------------------------------------------------
# Software Installation:
### Step-1: Download the H2HI_WiFi.zip file in your PC from here: [https://zenodo.org/record/7878129](https://zenodo.org/record/7878129) . Then, extract the *.zip file.
### Step-2: Python Requirement
 - Download 'Windows installer (64-bit)' of the 'python 3.9.7' from this link: [https://www.python.org/downloads/release/python-397/](https://www.python.org/downloads/release/python-397/) and install in your pc.
### Step-3: Tensorflow-GPU Requirement
 - I recommend using the H2HI_WiFi software using external NVIDIA GPU since it makes the computation much faster. If your pc doesn't have GPU, it will still work slowly without GPU. But if your PC has GPU, I recommend installing the CUDNN dependencies by following:
    1. From 'H2HI_WiFi\Read Me\Tensorflow GPU dependencies' folder, install 'Visual Studio 2019 v16.11.8.exe' and then Sign In to your Microsoft account using Visual Studio window. You need to have your internet connection turned on since it downloads the software files.
    2. From 'H2HI_WiFi\Read Me\Tensorflow GPU dependencies' folder, install 'cuda_11.2.2_win10_network.exe'. You need to have your internet connection turned on since it downloads the software files. You may watch the video regarding this installation from here: [https://youtu.be/hHWkvEcDBO0?t=177](https://youtu.be/hHWkvEcDBO0?t=177)
    3. From 'H2HI_WiFi\Read Me\Tensorflow GPU dependencies' folder, Copy-Paste files from the 'cudnn-11.2-windows-x64-v8.1.1.33.zip' file according to the Youtube video: [https://youtu.be/hHWkvEcDBO0?t=234](https://youtu.be/hHWkvEcDBO0?t=234)
    4. Follow the steps of the youtube video starting from 4:40 minute until 5:36 minute : [https://youtu.be/hHWkvEcDBO0?t=280](https://youtu.be/hHWkvEcDBO0?t=280)
    5. Restart your PC.

### Step-4: Create Virtual Environment, Activate it, Install Libraries
- If you don't know how to create a virtual environment and install libraries from 'requirements.txt', I recommend watching this video first: [https://youtu.be/9a1NDDcDQ7c](https://youtu.be/9a1NDDcDQ7c)
- Now, while you are in the 'H2HI_WiFi' folder, click on the 'Address Bar', write 'cmd' and press 'Enter' button.
- On the command prompt window, paste and run the following codes:
```
    python -m venv venv
    venv\scripts\activate
    pip install -r requirements.txt
```
----------------------------------------------------------------------------------------------------
# Run the software: 
- Double click on 'H2HI_WiFi.exe' and use it.
----------------------------------------------------------------------------------------------------

# Software Constraints:
1. Minimum Wifi packets suppressed to 1560. More than 1560 packets will be reduced into 1560 and less than that will be padded with 0 values to make the length 1560.
2. During data preprocessing, clipping of extra WiFi packets is done from steady-state portion.
   From Table-3 of Dataset paper (https://www.sciencedirect.com/science/article/pii/S235234092030562X#tbl0001), it's observed that, 
   Approaching (I1) interaction has 'Steady State' at the end and other interactions has 'Steady State' at first.
   Since we've to clip the WiFi packets to number of min_packets, we'll perform it from 'Steady State'.
   Hence during clipping, file name was checked during training to identify I1 interaction.
   During test, this is not possible; hence it's a must to complete interaction within
   1560 WiFi packets. Otherwise, keep '_I1_T' portion in file name of Approaching (I1) interaction
   and don't keep '_I1_T' portion in filename of other interactions.

3. App Window size fixed to 1920x1080. Resizing couldn't be done due to Grip Size problem.
4. It's advised to have GPU for faster processing, otherwise execution will be slower.
5. For the first-time run of this app, you've to be connected to internet for downloading the requirements of Plotly HTML plot. From next-time, you don't need to be online. If anytime you observe blank plot (white-screen), please enable internet connection and run the app.
