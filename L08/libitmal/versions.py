#!/opt/anaconda3/bin/python

def Versions():    
    import sys    
    print(f'{"Python version:":28s} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.')
    
    try:
        import sklearn as version_skl 
        print(f'{"Scikit-learn version:":28s} {version_skl.__version__}.')
    except:
        print(f'WARN: could not find sklearn!')  
    try:
        import keras as version_kr
        print(f'{"Keras version:":28s} {version_kr.__version__}')
    except:
        print(f'WARN: could not find keras!')  
    try:
        import tensorflow as version_tf
        print(f'{"Tensorflow version:":28s} {version_tf.__version__}')
    except:
        print(f'WARN: could not find tensorflow!')  
    try:
        import tensorflow.keras as version_tf_kr
        print(f'{"Tensorflow.keras version:":28s} {version_tf_kr.__version__}')
    except:
        print(f'WARN: could not find tensorflow.keras!')  

def TestAll():
	Versions()
	print("ALL OK")

if __name__ == '__main__':
	TestAll()