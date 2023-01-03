#### Reagglomeration of PyChunkedgGraph local affinity graph of supervoxels

Designed to work with the output of CAVEClient

example usage:

```
from agglomerate import agglomerate_graph
import pandas as pd
import caveclient
import numpy as np

client = caveclient.CAVEclient('v1dd', server_address='https://global.em.brain.allentech.org/')

cv=client.info.segmentation_cloudvolume()
position_viewer= np.array([63112, 94800, 9561])
position = position_viewer*(np.array([9,9,45])/[9.7,9.7,45.0])
position=position.astype(np.int32)
sv_id=cv.download_point(position_viewer, size=1,
                        coord_resolution=client.info.viewer_resolution(), 
                        agglomerate=False)
sv_id=int(np.squeeze(sv_id))
root_id = client.chunkedgraph.get_root_id(sv_id)

width = cv.chunk_size
bounds = [position-width.tolist(), position+width.tolist()]
bounds = np.array(bounds, dtype=np.int32)

edges, affinities, areas = client.chunkedgraph.get_subgraph(root_id, bounds.T)

agg_df = agglomerate_graph(edges, affinities, areas, 0)
```
