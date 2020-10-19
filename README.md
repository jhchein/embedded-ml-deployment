**work in progress**

# Deploy models on microcontrollers
Train Tensorflow Lite Model and convert to C++ code in order to run eventually on a microcontroller

# Build Dockerfile and push to ACR
* az login
* az acr login --name *registryname*
* az acr build --image *imagename* --registry *registryname* --file Dockerfile .
