#!/bin/bash
app="dockerize_tesis"
docker build -t ${app} .
docker run -d -p 56733:80 --name tesis -v "$PWD":/app ${app}