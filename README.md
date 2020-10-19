# embedded-ml-deployment
Train Python Model and compile to C++ to run models on microcontrollers.

We will start with implementing TF Lite for microcontroller examples into a Azure DevOps training pipeline.

# Build Dockerfile and push to ACR
az login
az acr login --name *registryname*
az acr build --image *imagename* --registry *registryname* --file Dockerfile .
