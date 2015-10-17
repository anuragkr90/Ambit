SETUP Steps:
      1. Python packages to be installed
      	 numpy, scipy, sklearn 
      2. Install mfcc_features
      	 Inside mfcc_features run -- python setup.py install
	 may have to do sudo 
      3. Libsvm --
      	 Nothing needs to be done here unless there's an error (unlikely)
	 If you are getting some sort of error with libsvm
	    1. Download libsvm from here https://www.csie.ntu.edu.tw/~cjlin/libsvm/ 
	    2. Compile (for python) for your system (details in the package)
	    3. Substitute the current libsvm.so.2 with your own


RUN
    python getDetRes.py "wavfile name"
    Eg. python getDetRes.py MarkkitSessionA_1200m_00s__1320m_00s_clip22.wav
    
    It will show the detection results as
    [Detetion_Results(0/1) Segment_Start_Time(in seconds) Segment_End_Time(in Seconds)]
    It will also dump this output in a file. Change Output type or form as needed. 



    Due to some changes in the training/test the latency has increased a little bit. But it should be good enough for Demo
