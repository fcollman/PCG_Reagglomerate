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
position = position_viewer*(client.info.viewer_resolution()/cv.resolution)
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

agg_df.head(5) will be 

|    |                v0 |                v1 |   merged_affinity |   merged_area |
|---:|------------------:|------------------:|------------------:|--------------:|
|  0 | 85165423116943360 | 85165491836423040 |               inf |             1 |
|  1 | 85094985653157920 | 85095054372679120 |               inf |             1 |
|  2 | 85024685628614368 | 85024616909111344 |               inf |             1 |
|  3 | 85024685628457040 | 85024616908964736 |               inf |             1 |
|  4 | 85095054372634768 | 85095123092094480 |               inf |             1 |

add_df.tail(5) will be

|       |                v0 |                v1 |   merged_affinity |   merged_area |
|------:|------------------:|------------------:|------------------:|--------------:|
| 73304 | 85024616909024816 | 85095123092096672 |                 6 |           110 |
| 73305 | 85024616909060944 | 85095123092096672 |                 6 |           210 |
| 73306 | 85095123092247440 | 85095123092096672 |                 5 |           110 |
| 73307 | 85024685628666048 | 85095123092096672 |                 4 |           110 |
| 73308 | 85095123091729472 | 85095123091810512 |                 3 |            20 |
