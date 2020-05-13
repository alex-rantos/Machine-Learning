def setBoxColors(bp):
   bp['boxes'][0].set( color='b',facecolor='b')
   bp['boxes'][1].set( color='r',facecolor='r' )

   bp['whiskers'][0].set( color='b')
   bp['whiskers'][1].set( color='b')
   bp['whiskers'][2].set( color='r')
   bp['whiskers'][3].set( color='r')

   bp['caps'][0].set( color='b')
   bp['caps'][1].set( color='b')
   bp['caps'][2].set( color='r')
   bp['caps'][3].set( color='r')

    
   bp['medians'][0].set( color='b')
   bp['medians'][1].set( color='r')

   bp['fliers'][0].set( color='b')
   bp['fliers'][1].set( color='r')
    
#ftiaxnoume ta dataframes mono me numerical values
cols_to_transform.append('Label')
cols_to_transform.append('Id')
box_good=good.drop(cols_to_transform, axis = 1)
box_bad=bad.drop(cols_to_transform, axis =1)
for x in box_good.columns:
  fig,ax = plt.subplots()
  data=[box_good[x],box_bad[x]]
  bp=ax.boxplot(data,positions = [1, 2], widths = 0.6,patch_artist=True)
  setBoxColors(bp)
  ax.set_title(x)
  ax.set_xlabel('Distribution')
  ax.set_ylabel('Values')
  ax.set_xticklabels(['Good','Bad'])
  #ax.set_xticklabels(range(10))
  plt.show()      
  fig.savefig('Numerical Visualization %s' % x)
