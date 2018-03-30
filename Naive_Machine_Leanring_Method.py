import sklearn
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BaseNB,GaussianNB,BernoulliNB,MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Naive_Machine_Learning_Method:
    """Here some naive machine learning methods to finish classification task.
        INPUT:
            train_feature: a ndarray of the feature, each row belongs to one train data.
            train_label: a list of train datas' labels.
            valid_feature: a ndarray of the feature, each row belongs to one valid data.
            valid_label: a list of valid datas' labels. 
                        If you want to generate valid label, you don't have to input valid_label.
            mode: There three modes--'Tuning','Valid' and 'Test'. In 'Tuning' mode, it will find 
                the best parameter for the train data. In 'Valid' mode, it will generate label for
                the valid data and compared with the true label to see whether the model works. In
                'Test' mode, it will generate label for the valid data.
            method: a list of methods you want to choose.
        OUTPUT:
            If you choose 'Test' mode, it will return the predicted label for the valid set.
            Otherwise it will return nothing.
    """
    
    def __init__(self, train_data, train_label, valid_data, valid_label, mode,
                 method=['LiR','GNB','BNB','MNB','LR','P','DT','KNN','RF','SVM_rbf','SVM_linear']):
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.mode = mode
        self.method =method     
        #self.GridSearchCV = GridSearchCV(n_jobs=5,verbose=1,cv=5,scoring='accuracy')
        if self.mode == 'Test' or self.mode == 'Valid':
            self.my_GNB = GaussianNB()
            self.my_BNB = BernoulliNB()
            self.my_MNB = MultinomialNB()
            self.my_LiR = LinearRegression(normalize=True)
            self.my_LR = LogisticRegression(solver='liblinear') 
            self.my_P = Perceptron(max_iter=100000,tol=0.1)
            self.my_DT = DecisionTreeClassifier()
            self.my_RF = RandomForestClassifier(criterion='entropy')
            self.my_KNN = KNeighborsClassifier()
            self.my_SVM_linear = SVC(kernel='linear',max_iter=100000)
            self.my_SVM_rbf = SVC(kernel='rbf', max_iter=100000)
        if 'NB' in self.method:     self.method_BaseNB()
        if 'GNB' in self.method:    self.method_GaussianNB()
        if 'BNB' in self.method:    self.method_BernoulliNB()
        if 'MNB' in self.method:    self.method_MultinomialNB()
        if 'LiR' in self.method:     self.method_LinearRegression()
        if 'LR' in self.method:     self.method_LogisticRegression()
        if 'P' in self.method:      self.method_Perception()
        if 'DT' in self.method:     self.method_DecisionTreeClassifier()
        if 'KNN' in self.method:    self.method_KNeighborsClassifier()
        if 'RF' in self.method:     self.method_RandomForestClassifier()
        if 'SVM_rbf' in self.method:    self.method_SVM_rbf()
        if 'SVM_linear' in self.method: self.method_SVM_linear()
            
        
    def method_GaussianNB(self):
        start = time.clock()
        self.my_GNB.fit(self.train_data, self.train_label)
        self.my_GNB_pred = self.my_GNB.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_GNB_acc = accuracy_score(self.my_GNB_pred, self.valid_label)
            print('GaussianNB accuracy is: ' + str(self.my_GNB_acc))  
        if self.mode == 'Test':
            return(self.my_GNB_pred)
        end =time.clock()
        print('GaussianNB method cost '+str(end-start)+' s.')  
        
    def method_BernoulliNB(self):
        start = time.clock()
        self.my_BNB.fit(self.train_data, self.train_label)
        self.my_BNB_pred = self.my_BNB.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_BNB_acc = accuracy_score(self.my_BNB_pred, self.valid_label)
            print('BernoulliNB accuracy is: ' + str(self.my_BNB_acc))  
        if self.mode == 'Test':
            return(self.my_BNB_pred)
        end =time.clock()
        print('BernoulliNB method cost '+str(end-start)+' s.')  
        
    def method_MultinomialNB(self):
        start = time.clock()
        self.my_MNB.fit(self.train_data, self.train_label)
        self.my_MNB_pred = self.my_MNB.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_MNB_acc = accuracy_score(self.my_MNB_pred, self.valid_label)
            print('MultinomialNB accuracy is: ' + str(self.my_MNB_acc))  
        if self.mode == 'Test':
            return(self.my_MNB_pred)
        end =time.clock()
        print('MultinomialNB method cost '+str(end-start)+' s.')  
    
    def method_LinearRegression(self):
        start = time.clock()
        self.my_LiR.fit(self.train_data, self.train_label)
        self.my_LiR_pred = self.my_LiR.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_LiR_acc = accuracy_score(self.my_LiR_pred, self.valid_label)
            print('LinearRegression accuracy is: ' + str(self.my_LiR_acc))
        if self.mode == 'Test':
            return(self.my_LiR_pred)
        end =time.clock()
        print('LinearRegression method cost '+str(end-start)+' s.')  
        
    def method_LogisticRegression(self):
        #solver=‘liblinear’ for small data, 'sag' and 'saga' for big; 
        #‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ for multi-classification. 
        #multi_class : str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
        start = time.clock()
        self.my_LR.fit(self.train_data, self.train_label)
        self.my_LR_pred = self.my_LR.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_LR_acc = accuracy_score(self.my_LR_pred, self.valid_label)
            print('LogisticRegression accuracy is: ' + str(self.my_LR_acc))
        if self.mode == 'Test':
            return(self.my_LR_pred)
        end =time.clock()
        print('LogisticRegression method cost '+str(end-start)+' s.')    
        
    def method_Perception(self):
        start = time.clock()
        self.my_P.fit(self.train_data,self.train_label)
        self.my_P_pred = self.my_P.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_P_acc = accuracy_score(self.my_P_pred, self.valid_label)
            print('Perception accuracy is: ' + str(self.my_P_acc))
        if self.mode == 'Test':
            return(self.my_P_pred)
        end =time.clock()
        print('Perception method cost '+str(end-start)+' s.')    
        
    def method_DecisionTreeClassifier(self):
        start = time.clock()
        self.my_DT.fit(self.train_data, self.train_label)
        self.my_DT_pred = self.my_DT.predict(self.valid_data)
        if self.mode == 'Valid':
            self.my_DT_acc = accuracy_score(self.my_DT_pred, self.valid_label)
            print('DecisionTreeClassifier accuracy is: ' + str(self.my_DT_acc))
        if self.mode == 'Test':
            return(self.my_DT_pred)
        end =time.clock()
        print('DecisionTreeClassifier method cost '+str(end-start)+' s.')   
        
    def method_KNeighborsClassifier(self):
        start = time.clock()
        if self.mode == 'Tuning':
            pipeline = Pipeline([('clf', KNeighborsClassifier())])
            parameters = {'clf__n_neighbors': (5, 10, 3, 50)}
            grid_search = self.GridSearchCV(pipeline, parameters)
            grid_search.fit(self.train_data, self.train_label)
            print('Best score: %0.3f' % grid_search.best_score_)
            print('Best parameters; ')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(best_parameters.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]))
        else:
            self.my_KNN.fit(self.train_data, self.train_label)
            self.my_KNN_pred = self.my_KNN.predict(self.valid_data)
            if self.mode == 'Valid':
                self.my_KNN_acc = accuracy_score(self.my_KNN_pred, self.valid_label)
                print('KNeighborsClassifier accuracy is: ' + str(self.my_KNN_acc))
            if self.mode == 'Test':
                return(self.my_KNN_pred)
        end =time.clock()
        print('KNeighborsClassifier method cost '+str(end-start)+' s.')   
       
    def method_RandomForestClassifier(self):
        start = time.clock()
        if self.mode == 'Tuning':
            pipeline = Pipeline([('clf', RandomForestClassifier(criterion='entropy'))])
            parameters = {'clf__n_estimators': (5, 10, 20, 50),
                          'clf__max_depth': (50, 150, 250),
                          'clf__min_samples_split': (1.0, 2, 3),
                          'clf__min_samples_leaf': (1, 2, 3)}
            grid_search = self.GridSearchCV(pipeline, parameters)
            grid_search.fit(self.train_data, self.train_label)
            print('Best score: %0.3f' % grid_search.best_score_)
            print('Best parameters; ')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(best_parameters.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]))
        else:
            self.my_RF.fit(self.train_data, self.train_label)
            self.my_RF_pred = self.my_RF.predict(self.valid_data)
            if self.mode == 'Valid':
                self.my_RF_acc = accuracy_score(self.my_RF_pred, self.valid_label)
                print('RandomForestClassifier accuracy is: ' + str(self.my_RF_acc))
            if self.mode == 'Test':
                return(self.my_RF_pred)
        end =time.clock()
        print('RandomForestClassifier method cost '+str(end-start)+' s.')   
               
    def method_SVM_rbf(self):
        start = time.clock()
        if self.mode == 'Tuning':
            pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100))])
            parameters = {'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),    
                          'clf__C': (0.1, 0.3, 1, 3, 10, 30), }
            grid_search = self.GridSearchCV(pipeline, parameters)
            grid_search.fit(self.train_data, self.train_label)
            print('Best score：%0.3f' % grid_search.best_score_)
            print('Best paragram：')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]))
        else:
            self.my_SVM_rbf.fit(self.train_data, self.train_label)
            self.my_SVM_rbf_pred = self.my_SVM_rbf.predict(self.valid_data)
            if self.mode == 'Valid':
                self.my_SVM_rbf_acc = accuracy_score(self.my_SVM_rbf_pred, self.valid_label)
                print('SVM(rbf kernel) accuracy is: ' + str(self.my_SVM_rbf_acc))
            if self.mode == 'Test':
                return(self.my_SVM_rbf_pred)
        end =time.clock()
        print('SVM(rbf kernel) method cost '+str(end-start)+' s.')   
        
    def method_SVM_linear(self):
        start = time.clock()
        if self.mode == 'Tuning':
            pipeline = Pipeline([('clf', SVC(kernel='linear', gamma=0.01, C=100,max_iter=100000))])
            parameters = {'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),    
                          'clf__C': (0.1, 0.3, 1, 3, 10, 30), }
            grid_search = self.GridSearchCV(pipeline, parameters)
            grid_search.fit(self.train_data, self.train_label)
            print('Best score：%0.3f' % grid_search.best_score_)
            print('Best paragram：')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]))
        else:
            self.my_SVM_linear.fit(self.train_data, self.train_label)
            self.my_SVM_linear_pred = self.my_SVM_linear.predict(self.valid_data)
            if self.mode == 'Valid':
                self.my_SVM_linear_acc = accuracy_score(self.my_SVM_linear_pred, self.valid_label)
                print('SVM_linear accuracy is: ' + str(self.my_SVM_linear_acc))
            if self.mode == 'Test':
                return(self.my_SVM_linear_pred)
        end =time.clock()
        print('SVM(linear kernel) method cost '+str(end-start)+' s.')   
     

if __name__ =='__main__':
    data = X
    #label = np.zeros((24,1))
    #for i in range(24):
    #    label[i] = valid[i]
    label = valid
    Naive_Machine_Learning_Method(data[:20], label[:20],data[20:],label[20:],mode='Valid',method=['LiR'])
