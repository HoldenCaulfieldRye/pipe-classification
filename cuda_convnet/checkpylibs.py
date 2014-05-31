if __name__ == "__main__":

    result = open('checkpylibs.txt', 'w')

    s = ""

    try:
        import joblib
        print "joblib is installed"
    except ImportError, e:
        s = "ain't got no joblib. "
        
    try:
        import sklearn
        print "scikit-learn is installed"
    except ImportError, e:
        s += "ain't got no sklearn."

    result.write(s)



