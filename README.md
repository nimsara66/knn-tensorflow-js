## KNN using tensorflow js

## Predict housing price according to longitude and latitude

# 
    1. features --> longitude and latitude
    2. labels --> house price
    3. k --> number of neighboures considered
    4. pp --> predictionPoint

#
    1. get the distance from the pp ( s = (x**2 + y**2)**0.5 )
    2. relate distance to house price
    3. sort the tensors relative to the pp
    4. get average of the top 'k' prices

[x] test with https://stephengrider.github.io/JSPlaygrounds/, dummy data
[x] simple workflow on the playground
[] migrate to vscode