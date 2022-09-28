#---------------------------------------------------------------------------------------------------------------
debug=True # set to False for 0 verbose
def traces():
    """debugging assist"""
    from inspect import currentframe, getouterframes, getframeinfo
    if debug:
        print("Stack trace")
        print("%15s"%"framename",' : lineno')
        frms=getouterframes(currentframe().f_back)
        for frm in frms:
            print("%15s"%frm.function,' : ',frm.lineno)
            if frm.function=='<module>':
                break
#---------------------------------------------------------------------------------------------------------------
def nulsCount(df):
    """summarise missing/unexpected values"""
    from pandas import DataFrame
    
    d2=DataFrame(columns=["NULL","NAN","BLANKS","UNEXP"])
    try:
        d2["NULL"] = df.isnull().sum().astype('uint32') # check for null values
        d2["NAN"]=df.isna().sum().astype('uint32') # check for NaN
        d2["BLANKS"]=df.isin([""," "]).sum().astype('uint32') # check for blanks
        d2["UNEXP"]=df.isin(["-","?",".","NA","N/A","nan","NAN","Nan","Unknown","UNKNOWN","unknown"]).sum().astype('uint32') # check for other unexpected values
    except:
        pass
    d2=d2.loc[(d2["NULL"]!=0) | (d2["NAN"]!=0) | (d2["BLANKS"]!=0) | (d2["UNEXP"]!=0)] # shortlist for the missing values
    
    # convert to percentages
    d2["NULL %"]=d2["NULL"].apply(lambda x: round(x*100/df.shape[0],2))
    d2["NAN %"]=d2["NAN"].apply(lambda x: round(x*100/df.shape[0],2))
    d2["BLANKS %"]=d2["BLANKS"].apply(lambda x: round(x*100/df.shape[0],2))
    d2["UNEXP %"]=d2["UNEXP"].apply(lambda x: round(x*100/df.shape[0],2))
    
    # rearrange
    d2=d2[["NULL","NULL %","NAN","NAN %","BLANKS","BLANKS %","UNEXP","UNEXP %"]]
    
    if d2.shape[0]==0:
        print('good to go')
        return
    else:     
        return d2
#---------------------------------------------------------------------------------------------------------------
class dummies():
    """to implement encoding without data leak"""
    def __init__(self):
        """input : dataframe"""
        self.ref={}
        self.fitted=False
    
    def fit(self,df):
        """Collect required encoding information"""
        cat=list(df.select_dtypes(include='object').columns)
        for col in cat:
            unq=list(df[col].value_counts().index)
            self.ref.update({col:unq})
        self.fitted=True
        return
    
    def transform(self,df):
        """perform encoding"""
        from pandas import to_numeric
        df=df.copy()
        if not self.fitted:
            raise ValueError("please fit first")
            return
        cat=list(self.ref.keys())
        for col in cat:
            unq=self.ref.get(col)
            for i in unq:
                df[col+"_"+str(i)]=df[col]
                df.loc[df[col+"_"+str(i)]==i,[col+"_"+str(i)]]=1
                df.loc[df[col+"_"+str(i)]!=1,[col+"_"+str(i)]]=0
            df.drop(col,axis=1,inplace=True)
            df.drop(col+"_"+str(unq[i]),axis=1,inplace=True) #drop_first=True
        for col in df.select_dtypes(exclude=[int,float]).columns:
            df[col]=df[col].astype('float',errors='ignore') # try casting non numeric features to float
        return df
    
    def fit_transform(self,df):
        """learn and encode"""
        self.fit(df)
        df=self.transform(df)
        return df
#---------------------------------------------------------------------------------------------------------------
class remap():
    def __init__(self):
        """performs skew correction and z-score standardisation"""
        self.fitted=False
        
    def fit(self,df,cols=None):
        """registers stats of the dataframe"""
        from pandas import DataFrame
        from numpy import log, sqrt, abs
        
        self.cols=cols
        if self.cols==None:
            self.cols=df.columns
        df=df[self.cols].copy()
        
        self.fitting_info=DataFrame(columns=["skew","kurt","min","max","reflect","r_min","r_max","mms","log","sqrt"],
                                       index=df.columns)
        
        # initialise flags
        self.fitting_info["reflect"] = False
        self.fitting_info["mms"] = False
        self.fitting_info["log"] = False
        self.fitting_info["sqrt"] = False
        
        # reocird basic stats
        self.fitting_info["skew"] = df.skew()
        self.fitting_info["kurt"] = df.kurt()
        self.fitting_info["min"] = df.min()
        self.fitting_info["max"] = df.max()
        
        # test need for reflected transforms
        collist=list(self.fitting_info.loc[self.fitting_info["skew"]<=-0.75].index)
        for col in collist:            
            # read basic stats
            [cskew,cmin,cmax]=self.fitting_info.loc[col,["skew","min","max"]]
            
            # reflect
            temp_r = cmax+1-df[col]
            cmin=temp_r.min()
            cmax=temp_r.max()
            self.fitting_info.loc[col,["r_min","r_max"]]=[cmin,cmax]
            # scale between 0-500
            temp_r_mms = (temp_r-cmin)*500/(cmax-cmin)
            self.fitting_info.loc[col,["mms_min","mms_max"]]=[temp_r_mms.min(),temp_r_mms.max()]
            # scaled log tranform
            temp_r_mms_l = (temp_r_mms+1).apply(log)
            # scaled sqrt tranform
            temp_r_mms_s = temp_r_mms.apply(sqrt)
            # plain log tranform
            temp_r_l = (temp_r+1).apply(log)
            # plain sqrt tranform
            temp_r_s = temp_r.apply(sqrt)
            # transformed skews
            t_skew = abs([temp_r_l.skew(),temp_r_s.skew(),temp_r_mms_l.skew(),temp_r_mms_s.skew()])
            # register flags
            if round(min(t_skew),2)<round(abs(cskew),2):
                self.fitting_info.loc[col,"reflect"]=True
                if min(t_skew)==t_skew[0]:
                    self.fitting_info.loc[col,"log"]=True
                    df[col]=temp_r_l
                elif min(t_skew)==t_skew[1]:
                    self.fitting_info.loc[col,"sqrt"]=True
                    df[col]=temp_r_s
                elif min(t_skew)==t_skew[2]:
                    self.fitting_info.loc[col,["log","mms"]]=[True,True]
                    df[col]=temp_r_mms_l
                elif min(t_skew)==t_skew[3]:
                    self.fitting_info.loc[col,["sqrt","mms"]]=[True,True]
                    df[col]=temp_r_mms_s                
        
        # test need for plain transforms
        collist=list(self.fitting_info.loc[self.fitting_info["skew"]>=0.75].index)
        for col in collist:            
            # read basic stats
            [cskew,cmin,cmax]=self.fitting_info.loc[col,["skew","min","max"]]
            
            # scale between 0-500
            temp_mms = (df[col]-cmin)*500/(cmax-cmin)
            self.fitting_info.loc[col,["mms_min","mms_max"]]=[temp_mms.min(),temp_mms.max()]
            # scaled log tranform
            temp_mms_l = (temp_mms+1).apply(log)
            # scaled sqrt tranform
            temp_mms_s = temp_mms.apply(sqrt)
            # plain log tranform
            temp_l = (df[col]+1).apply(log)
            # plain sqrt tranform
            temp_s = df[col].apply(sqrt)
            # transformed skews
            t_skew = abs([temp_l.skew(),temp_s.skew(),temp_mms_l.skew(),temp_mms_s.skew()])
            # register flags
            if round(min(t_skew),2)<round(abs(cskew),2):
                if min(t_skew)==t_skew[0]:
                    self.fitting_info.loc[col,"log"]=True
                    df[col]=temp_l
                elif min(t_skew)==t_skew[1]:
                    self.fitting_info.loc[col,"sqrt"]=True
                    df[col]=temp_s
                elif min(t_skew)==t_skew[2]:
                    self.fitting_info.loc[col,["log","mms"]]=True
                    df[col]=temp_mms_l
                elif min(t_skew)==t_skew[3]:
                    self.fitting_info.loc[col,["sqrt","mms"]]=[True,True]
                    df[col]=temp_mms_s
        
        # set fitted flag
        self.fitted=True             
    
    def transform(self,df):
        """perform transforms & scaling"""
        if not self.fitted:
            raise ValueError("please fit remap")
            return
        
        from pandas import merge
        from numpy import log, sqrt, abs
        
        df_orig=df.copy()
        df=df[self.cols].copy()
        
        for col in df.columns:            
            # find min max value
            cmin = self.fitting_info.loc[col,"min"]
            cmax = self.fitting_info.loc[col,"max"]
            
            # 1. reflection
            if self.fitting_info.loc[col,"reflect"]:
                temp = cmax+1-df[col]
                df[col] = temp
                # update min max
                cmin = self.fitting_info.loc[col,"r_min"] 
                cmax = self.fitting_info.loc[col,"r_max"]
                    
            # 2. min max scaling for log / sqrt
            if self.fitting_info.loc[col,"mms"]:
                temp = (df[col]-cmin)*500/(cmax-cmin)
                df[col] = temp
                # update min max
                cmin = self.fitting_info.loc[col,"mms_min"] 
                cmax = self.fitting_info.loc[col,"mms_max"]
            
            # 3. shift data to +ve scale
            if cmin<0:
                df[col]=df[col]-cmin 
            if df[col].min()<0: # reconfirm
                df[col]=df[col]-df[col].min()
                    
            # 4. log transform
            if self.fitting_info.loc[col,"log"]:
                df[col]=(df[col]+1).apply(log)
                
            # 5. sqrt transform
            if self.fitting_info.loc[col,"sqrt"]:
                df[col]=df[col].apply(sqrt)
                
            # 6. reverse Reflection
            if self.fitting_info.loc[col,"reflect"]:
                temp = log(cmax)+1-df[col]
                df[col] = temp
            
            # find skew
            self.fitting_info.loc[col,"trans_skew"]=df[col].skew()
        
        # find scaled skew
        self.fitting_info["trans_scaled_skew"]=df.skew()
            
        df_orig.drop(self.cols,axis=1,inplace=True)
        df=merge(df_orig,df,left_index=True,right_index=True,how='inner')
        
        return df
    
    def fit_transform(self,df):
        """fit, remap"""
        self.fit(df)
        df=self.transform(df)
        return df
#---------------------------------------------------------------------------------------------------------------
class pandaPoly():
    """PolynomialFeatures extraction and returns Pandas DataFrame"""
    
    def __init__(self,degree=2, interaction_only=False, include_bias=False):
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree, interaction_only, include_bias)
        self.fitted=False
    
    def fit(self,df):
        self.poly.fit(df)
        self.fitted=True
    
    def transform(self,df):
        if self.fitted:
            from pandas import DataFrame, merge
            df=df.copy()
            d2=DataFrame(self.poly.transform(df),index=df.index)
            d2=merge(df,d2,left_index=True,right_index=True)
            return d2
        else:
            raise ValueError("please fit pandaPoly")
    
    def fit_transform(self,df):
        self.fit(df)
        df=self.transform(df)
        return df     
#---------------------------------------------------------------------------------------------------------------
class pandaCluster():
    """performs KMeans Clustering and returnd Pandas DataFrame with cluster encoded columns"""
    
    def __init__(self):
        
        self.fitted=False
        from sklearn.preprocessing import StandardScaler
        
        # models
        self.scl = StandardScaler()
        self.dum = dummies()
        
    def clusterRank(self,df,max_clusters=10):
        
        from pandas import DataFrame
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
    
        clusters = range(2,max_clusters)
        self.elbow=DataFrame(columns=['n_clusters','distortion','slope','slope_delta'])
        self.best=DataFrame(columns=['rank','n_clusters','distortion','slope_delta'],index=[1,2])
        meanDistortions=[]

        # run clustering and measure distortions
        for k in clusters:
            model=KMeans(n_clusters=k)
            model.fit(df)
            prediction=model.predict(df)
            meanDistortions.append(sum(np.min(cdist(df, model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

        # analyse change in distortions
        slope=[]
        # slope of graph
        slope.extend([meanDistortions[i+1]-meanDistortions[i] for i in range(len(meanDistortions)-1)])
        slope.append(np.nan)

        slope_delta=[np.nan]
        # change of slope of graph
        slope_delta.extend([slope[i]-slope[i-1] for i in range(1,len(slope)-1)])
        slope_delta.append(np.nan)

        self.elbow.n_clusters=clusters
        self.elbow.distortion=meanDistortions
        self.elbow.slope=slope
        self.elbow.slope_delta=slope_delta
        self.elbow=self.elbow.sort_values(by=['slope_delta','slope'],ascending=False)

        # rank number of cluster based on change of slope
        for i in range(1,3):
            set1=[]
            ind=self.elbow.index[i-1]
            set1.append(i)
            set1.append(ind+2)
            set1.append(self.elbow['distortion'].loc[ind])
            set1.append(self.elbow['slope_delta'].loc[ind])
            self.best.loc[i]=set1

        # visualise
        import plotly.graph_objects as go
        gdata=self.elbow.sort_index().copy()
        fig=go.Figure()
        # plot optimal cluster numbers
        fig.add_trace(go.Scatter(x=self.best.n_clusters, y=self.best.distortion,
                                 mode='markers',name='optimal clusters',
                                 marker={'size':15,'color':'#FFA15A'},
                                 text=self.best['rank'],
                                 hovertemplate='<b>OPTIMA %{text}: %{x} clusters)'))
        # plot the distortions for cluster range of 2-9
        fig.add_trace(go.Scatter(x=gdata.n_clusters, y=gdata.distortion,name='Distortions',
                                 marker={'color':'#1F77B4'},
                                 hovertemplate='<b>Distortions</b><br>'+
                                 'n_clusters: %{x}<br>'+
                                 'distortion: %{y:.2f}'))
        fig.update_xaxes(title_text="n_clusters")
        fig.update_yaxes(title_text="distortions")
        fig.update_layout(title="Selecting k with the Elbow Method")
        fig.show()
    
    def fit(self,df):
        from pandas import DataFrame
        from sklearn.cluster import KMeans
        
        df=df.copy()
        
        # scale the data
        df= DataFrame(self.scl.fit_transform(df),columns=df.columns,index=df.index)
        # optimal cluster choice
        self.clusterRank(df)
        n=self.best.loc[self.best['rank']==2,'n_clusters'].values[0]
        # cluster fitting
        self.clt = KMeans(n_clusters=n)
        self.clt.fit(df)
        # encoder fitting for clusters
        pred=DataFrame(self.clt.predict(df),columns=["CLUSTER"],index=df.index,dtype='object')
        self.dum.fit(pred)
        
        self.fitted=True
    
    def transform(self,df):
        if self.fitted:
            from pandas import DataFrame, merge
            df=df.copy()
            dforig=df.copy()
            # scale the data
            df=DataFrame(self.scl.transform(df),columns=df.columns,index=df.index)
            # predict clusters
            pred=DataFrame(self.clt.predict(df),columns=["CLUSTER"],index=df.index,dtype='object')
            # encode cluster columns
            pred=self.dum.transform(pred)
            # merge with source
            df=merge(dforig,pred,left_index=True,right_index=True)
            return df
        else:
            raise ValueError("please fit pandaCluster")
    
    def fit_transform(self,df):
        self.fit(df)
        df=self.transform(df)
        return df
#---------------------------------------------------------------------------------------------------------------
def cvSplitter(X,Y,k=10,seed=129):
    """Splits K folds and returns array of copied dataframes"""
    X=X.copy()
    Y=Y.copy()
    L=X.shape[0]
    from numpy import random, floor
    
    # seed pseudo random generator
    random.seed(seed)
    indices=random.choice(X.index,L,False)
    sets=[(int(floor(L*(i)/k)),int(floor(L*(i+1)/k))) for i in range(k)]
    Xtrains=[]
    Xvals=[]
    Ytrains=[]
    Yvals=[]
    ss=0
    for i in range(k):
        se=int(floor(L*(i+1)/k))
        Xvals.append(X.loc[list(indices[ss:se])].copy())
        Yvals.append(Y.loc[list(indices[ss:se])].copy())
        Xtrains.append(X.loc[list(indices[[j not in indices[ss:se] for j in indices]])].copy())
        Ytrains.append(Y.loc[list(indices[[j not in indices[ss:se] for j in indices]])].copy())
        ss=se
    return Xtrains,Ytrains,Xvals,Yvals
#---------------------------------------------------------------------------------------------------------------
class SCFS():
    """https://www.frontiersin.org/articles/10.3389/fgene.2021.684100/full
    Reference article for feature scoring
    SCFS (Standard deviation and Cosine similarity based Feature Selection)
    Credits to: Juanying Xie, Mingzhao Wang, Shengquan Xu, Zhao Huang and Philip W. Grant"""
    
    def __init__(self,kind='exp',threshold='auto'):
        """kind = {'exp','reciprocal','anti-similarity'} default='exp'
        threshold = {'auto', float between 0.0 and 1.0} default='auto'"""
        
        self.kind=kind
        self.verbose=1
        
        autoThresh={'exp':0.2,'reciprocal':0,'anti-similarity':0.2}
        if threshold=='auto':
            self.threshold=autoThresh.get(kind)
        else:
            self.threshold=threshold

        self.fitted=False
        
    def discernibility(self):
        """list down the feature discernibility
        same as sample standard deviations"""
        import numpy as np
        import pandas as pd
        m=self.df.shape[0]
        self.dis=[np.sqrt(sum((self.df[i]-sum(self.df[i])/m)**2)/(m-1)) for i in self.df.columns]
        self.dis=pd.Series(self.dis,index=self.df.columns,dtype=float)
    
    def cosineSimilarity(self):
        """populate the cosine similarities (absolute)"""
        import numpy as np
        import pandas as pd
        self.cosdf=pd.DataFrame(columns=self.df.columns,index=self.df.columns)
        for i in self.df.columns:
            for j in self.df.columns:
                norm_i=np.sqrt(self.df[i].dot(self.df[i]))
                norm_j=np.sqrt(self.df[j].dot(self.df[j]))
                self.cosdf.loc[i,j] = (np.abs(self.df[i].dot(self.df[j])))/(norm_i*norm_j)
                
    def independence(self):
        """evaluate the feature independance"""
        import numpy as np
        import pandas as pd
        
        dismaxarg=self.dis.index[np.argmax(self.dis)]
        self.ind=pd.Series(index=self.df.columns,dtype=float)

        for i in self.df.columns:
            if self.dis[i] == self.dis[dismaxarg]: # for feature with max stddev
                if self.kind == 'exp':
                    self.ind[i] = np.exp(max(-self.cosdf.loc[i]))
                elif self.kind == 'reciprocal':
                    self.ind[i] = max(1/self.cosdf.loc[i])
                elif self.kind == 'anti-similarity':
                    self.ind[i] = max(1-self.cosdf.loc[i])
            else:
                if self.kind == 'exp':
                    self.ind[i] = np.exp(min(-self.cosdf.loc[i,self.dis[self.dis>self.dis[i]].index]))
                elif self.kind == 'reciprocal':
                    self.ind[i] = min(1/self.cosdf.loc[i,self.dis[self.dis>self.dis[i]].index])
                elif self.kind == 'anti-similarity':
                    self.ind[i] = min(1-self.cosdf.loc[i,self.dis[self.dis>self.dis[i]].index])
                    
    def fit(self,df):
        """evaluate feature scores of df"""
        self.df=df.copy()
        
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        
        mima=MinMaxScaler(feature_range=(-1,1))
        self.df = pd.DataFrame(mima.fit_transform(self.df),
                           columns=self.df.columns,index=self.df.index)
        
        self.discernibility()
        self.cosineSimilarity()
        self.independence()
        
        self.fscore=self.dis.mul(self.ind)
        self.fitted=True
        
        if self.verbose!=0:
            import plotly.graph_objects as go
            # lets review the feature scores
            fig=go.Figure()
            gdata=self.fscore.sort_values()
            fig.add_trace(go.Scatter(x=gdata.index, y=gdata,name='feature score'))
            fig.update_xaxes(title="features-->")
            fig.update_yaxes(title="scores-->")
            fig.show()

            fig=go.Figure()
            fig.add_trace(go.Scatter(x=scfs.dis,y=scfs.ind,mode='markers',name='discernibility vs independence'))
            fig.update_xaxes(title="discernibility-->")
            fig.update_yaxes(title="independence-->")
            fig.show()
        
    def fit_transform(self,df):
        """interatively reduce features"""
        import gc
        import numpy as np
        import pandas as pd
        
        self.verbose=0
        self.fit(df)        
        logscore=np.log(self.fscore)
        flag=logscore.min()
        
        while flag<self.threshold:
            gc.collect()
            len(gc.get_objects());
            
            ind=logscore.argmin()
            feat=self.fscore.index[ind]
            self.df.drop(feat,axis=1,inplace=True)
            self.fit(self.df)
            logscore=np.log(self.fscore)
            flag=logscore.min()
        
        return self.df
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# defining a function to report classification metrics
def classReporter(Y_train, pred_train, Y_test, pred_test,model_name):
    """Classification report
    logs test scores to global dataframe named scoreLog
    the scoreLog (with any previous scores) will be displayed
    also displays confusion matrices of current instance of arguments
    ---------------------------------------------------------------------------
    Y_train ==> TRUE classes used for training (pandas series object or numpy array of 1-D)
    pred_train ==> PREDICTION on training data (pandas series object or numpy array of 1-D)
    Y_test ==> TRUE classes to be used for testing (pandas series object or numpy array of 1-D)
    pred_test ==> PREDICTION on test data (pandas series object or numpy array of 1-D)
    model_name ==> str name for current model, to be used as index for scoreLog
    ---------------------------------------------------------------------------
    """
    from sklearn import metrics
    import plotly.figure_factory as ff
    import numpy as np
    import pandas as pd
    
    global scoreLog
    
    classes=list(Y_test.unique())
    cols=["accuracy"]
    cols.extend(["precision_"+str(classes[i]) for i in range(len(classes))])
    cols.extend(["recall_"+str(classes[i]) for i in range(len(classes))])
    cols.extend(["fscore_"+str(classes[i]) for i in range(len(classes))])
    
    try:
        type(scoreLog)
    except:
        scoreLog=pd.DataFrame(columns=cols)
    
    #metrics based on training set
    #confusion matrix
    z=pd.DataFrame(metrics.confusion_matrix(Y_train, pred_train))
    fig1=ff.create_annotated_heatmap(np.array(z),annotation_text=np.array(z),
                                    x=list(np.sort(np.unique(Y_train))),y=list(np.sort(np.unique(Y_train))),
                                    colorscale='Mint',font_colors = ['grey','white'],name="TRAINING SET",
                                    hovertemplate="Prediction: %{x:d}<br>True: %{y:d}<br>Count: %{z:d}")
    fig1.update_layout(height=350,width=350)
    fig1.update_xaxes(title_text="PREDICTED (TRAINING SET) - "+model_name)
    fig1.update_yaxes(title_text="TRUE",tickangle=270)
    
    #scores
    score=[metrics.accuracy_score(Y_train,pred_train)]
    score.extend(metrics.precision_score(Y_train,pred_train,labels=classes,average=None))
    score.extend(metrics.recall_score(Y_train,pred_train,labels=classes,average=None))
    score.extend(metrics.f1_score(Y_train,pred_train,labels=classes,average=None))
    scoreLog=scoreLog.append(pd.DataFrame(score,index=cols,columns=[model_name+"_training"]).T)
    
    #metrics based on test set
    #confusion matrix
    z=pd.DataFrame(metrics.confusion_matrix(Y_test, pred_test))
    fig2=ff.create_annotated_heatmap(np.array(z),annotation_text=np.array(z),
                                    x=list(np.sort(np.unique(Y_test))),y=list(np.sort(np.unique(Y_test))),
                                    colorscale='Mint',font_colors = ['grey','white'],name="TEST SET",
                                    hovertemplate="Prediction: %{x:d}<br>True: %{y:d}<br>Count: %{z:d}")
    fig2.update_layout(height=350,width=350)
    fig2.update_xaxes(title_text="PREDICTED (TEST SET) - "+model_name)
    fig2.update_yaxes(title_text="TRUE",tickangle=270)
    
    #scores
    score=[metrics.accuracy_score(Y_test,pred_test)]
    score.extend(metrics.precision_score(Y_test,pred_test,labels=classes,average=None))
    score.extend(metrics.recall_score(Y_test,pred_test,labels=classes,average=None))
    score.extend(metrics.f1_score(Y_test,pred_test,labels=classes,average=None))
    scoreLog=scoreLog.append(pd.DataFrame(score,index=cols,columns=[model_name+"_test"]).T)
    
    # merge both confusion matrix heatplots
    fig=make_subplots(rows=1,cols=2,horizontal_spacing=0.05)
    fig.add_trace(fig1.data[0],row=1,col=1)#,name="training data")
    fig.add_trace(fig2.data[0],row=1,col=2)#,name="test data")

    annot1 = list(fig1.layout.annotations)
    annot2 = list(fig2.layout.annotations)
    for k  in range(len(annot2)):
        annot2[k]['xref'] = 'x2'
        annot2[k]['yref'] = 'y2'
    fig.update_layout(annotations=annot1+annot2) 
    fig.layout.xaxis.update(fig1.layout.xaxis)
    fig.layout.yaxis.update(fig1.layout.yaxis)
    fig.layout.xaxis2.update(fig2.layout.xaxis)
    fig.layout.yaxis2.update(fig2.layout.yaxis)
    fig.layout.yaxis2.update({'title': {'text': ''}})
    
    display(scoreLog)
    fig.show()
