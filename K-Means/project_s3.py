import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None

headers = ['x', 'y']
df = pd.read_csv('s3.csv',  names = headers)
df.columns = headers

def doKMeans(data, clusters = 0):
  from sklearn.cluster import KMeans
  model = KMeans(clusters,random_state=5)
  model.fit(data)
  return model.cluster_centers_, model.labels_, model.inertia_


n_clusters = 15
iterations = 10
threshold = 0.95


print("Initial Cluster Configuration")
centroids, labels, inertia = doKMeans(df, n_clusters)
print("Cluster Centroids:")
print(centroids)
print("Inertia = ", inertia)

fig = plt.figure()
#axes = plt.gca()
#axes.set_xlim([59,67])
#axes.set_ylim([26,33])
#plt.title('JOENSUU')
#plt.xlabel('Latitude')
#plt.ylabel('Longitude')
ax = fig.add_subplot(111)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker='o', alpha=0.25, linewidths=1, s=10)
ax.scatter(centroids[:,0],centroids[:,1], marker="x", color='red', linewidths=1, s=100)
plt.show()

df['label'] = pd.Series(labels, index=df.index)

print("Iterations = ", iterations)
print("Threshold = ", threshold)


##### Algorithm   #################################################################

X = pd.DataFrame()
O = pd.DataFrame()

for it in range(iterations):
    X = pd.DataFrame()
    
    for cnum in range(len(centroids)):
        T = df[df.label == cnum]
        center = centroids[cnum]
        
        from math import sqrt
        T['dist'] = ( (T.iloc[:,0] - center[0])**2 + (T.iloc[:,1] - center[1])**2 ).apply(sqrt)
        d_max = max(T.dist)
        T['factor'] = T.dist/d_max
        
        Q = T[T.factor >= threshold]        
        
        T = T[T.factor < threshold]
        #print(T, center)
        X = X.append(T)        
        O = O.append(Q)
        
    df = X
    #df.reset_index(drop=True, inplace=True)
    
X.reset_index(drop=True, inplace=True)

#X.to_csv('result.csv', index_label = False, index = False, header = False)
###############################################################################

print()
print("Final Cluster Configuration")
centroids, labels, inertia = doKMeans(X, n_clusters)
print("Cluster Centroids:")
for f in centroids:
    print(f[:2])
print("Inertia = ", inertia)

fig = plt.figure()
#axes = plt.gca()
#axes.set_xlim([59,67])
#axes.set_ylim([26,33])
#plt.title('JOENSUU')
#plt.xlabel('Latitude')
#plt.ylabel('Longitude')
ax = fig.add_subplot(111)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='o', alpha=0.25, linewidths=1, s=10)
ax.scatter(centroids[:,0],centroids[:,1], marker="x", color='red', linewidths=1, s=100)
ax.scatter(O.iloc[:, 0], O.iloc[:, 1], marker='o', color=['green'], alpha=0.2, linewidths=1, s=10)
plt.show()
