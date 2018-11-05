

#%% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    out = 'C:\MachineLearning\Clustering\RF\\'

    np.random.seed(0)
    digits = pd.read_csv('C:\MachineLearning\Bug_TestData_Partitioned.csv')
    digitsX = digits[['InvoiceMonth1','InvoiceMonth2','InvoiceMonth3','InvoiceMonth4','InvoiceMonth5','InvoiceMonth6','InvoiceMonth7','InvoiceMonth8','InvoiceMonth9','InvoiceMonth10','InvoiceMonth11','InvoiceMonth12','PaymentMonth1','PaymentMonth2','PaymentMonth3','PaymentMonth4','PaymentMonth5','PaymentMonth6','PaymentMonth7','PaymentMonth8','PaymentMonth9','PaymentMonth10','PaymentMonth11','PaymentMonth12']].values
    digitsY = digits['Rating'].values

    madelon = pd.read_csv('C:\MachineLearning\Wine_TestData.csv')     
    madelonX = madelon[['Alcohol','MalicAcid','Ash','Alcalinity','Magnesium','TotalPhenols','Flavanoids','Nonflavanoids','Proanthocyanins','ColorIntensity','Hue','DilutionRatio','Proline']].values
    madelonY = madelon['Cultivator'].values

    
    
    madelonX = StandardScaler().fit_transform(madelonX)
    digitsX= StandardScaler().fit_transform(digitsX)
    
    clusters =  [2,5,10,15,20,25,30,35,40]
    dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
    
    #%% data for 1
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_madelon = rfc.fit(madelonX,madelonY).feature_importances_ 
    fs_digits = rfc.fit(digitsX,digitsY).feature_importances_ 
    print(fs_madelon)
    print(fs_digits)
    tmp = pd.Series(np.sort(fs_madelon)[::-1])
    tmp.to_csv(out+'madelon scree.csv')
    
    tmp = pd.Series(np.sort(fs_digits)[::-1])
    tmp.to_csv(out+'digits scree.csv')
    
    #%% Data for 2
    #dims = [2,3,4,5,7,10,13]
    filtr = ImportanceSelect(rfc)
    #grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    #mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    #pipe = Pipeline([('filter',filtr),('NN',mlp)])
    #gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    #gs.fit(madelonX,madelonY)
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv(out+'Madelon dim red.csv')
    
    dims = [2,3,4,5,10,15,20]
    grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}  
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(digitsX,digitsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'digits dim red.csv')
#    raise
    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    #dim = 3
    #filtr = ImportanceSelect(rfc,dim)
    
    #madelonX2 = filtr.fit_transform(madelonX,madelonY)
    #madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
    #cols = list(range(madelon2.shape[1]))
    #cols[-1] = 'Class'
    #madelon2.columns = cols
    #madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)
    
    dim = 2
    filtr = ImportanceSelect(rfc,dim)
    digitsX2 = filtr.fit_transform(digitsX,digitsY)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)