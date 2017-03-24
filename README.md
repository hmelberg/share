# share

Files:

'Model.ipynb': main file for running the model

'Evidence_synthesis.py': data collection, aggregation, and manipulation to be usable in model as well as reasoning behind choices.

'Inputs.py': what is actually used from 'Evidence_synthesis' at any version of the model

'Transitions': how agents transition between states in the model.

'Controls': widgets for main file

Background:

Hereditary breast – and ovarian cancer (HBOC) accounts for approximately 5–10% of annual incidences of these cancers in Norway, with the most frequent mutations found in the BRCA 1 and 2 genes. At present testing is either done by targeted analysis for the most frequently occurring BRCA mutations in the ethnic Norwegian population (founder mutations), or by complete sequencing of the BRCA-genes. Mutations in these genes only account for about 2% of annual incidences of breast – and ovarian cancer, and new sequencing techniques have recently made possible the parallel sequencing of many genes at relatively low marginal cost. 
The objective of this prediction model is to assess whether we are at a point where widening the search using multigene panels is a cost-effective alternative compared to both founder-mutation testing and complete BRCA sequencing. 
An important by-product of sequencing whole genes without having complete information about the pathogenicity of all possible finds is that the test result can come back inconclusive. Without a strong family history of cancer, identification of such variants of unknown significance is not recommended to trigger clinical action according to Norwegian guidelines. 
While the cost-effectiveness of testing for well-known mutations in the BRCA genes have previously been found favourable, these studies have not tackled the issue of inconclusive results when using whole-gene sequencing. This model explicitly incorporates this element, as well as the implications of cascade testing of family members of identified mutation-carriers that normally arises.  
