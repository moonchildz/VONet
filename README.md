README : VONet for reviewers 

This is codes for generate Virtual Organoids(VO), train VONet using VO, and test the performane of it. 

Since this code is for research, it is not automatically configured. 
I am sorry for not optimizing usage, since I am now doing lots of other works for changing my workplace.
I wrote how I used this code during my research. 
Please feel free to mail me when you feel difficulty while using. 

*VO sets used in Fig. S3 is stored in [dropbox link-VO used in Manuscript Figures]. However, VO sets used in Fig.S4 & S5 is not available, 
since we deleted the sets judging that they are useless. 

1. Train VONet

You can download image files of VO dataset from [dropbox link-VO dataset].  
Configure the directory of images and saving route of weight file in 
```dl_org_shape_v2.py```
and run the code. The model structure can be seen in 
``` net_org_v2.py```

2. Test VONet

You can download image files of RO dataset from [dropbox link-RO dataset].  
Since VONet is trained using only blue channel data of images, test also should be done using blue channel only. 
The test can be done by running 
```
ui_org.py
```
This is a tkinter powered file. 
Before using the file, please configure the weight file route in 
```
test_orgv3.py
```
