# Amt-Eval
This repository is aimed as a implementation agnostic amt metrics calculation tool.

## Initializtion of the repo
Create the Environment, creates env `amt-eval`. 
```shell
conda env create -f environment-current.yaml
```

* `environment-current.yaml` is the fixed-version environment file for a save env initialization. 
* `environment.yaml` is for upgrading -> fixed real dependencies.

```shell
conda env export > environment-gpu-current.yaml
```

## Testing
Running the tests:
```shell
cd tests && pytest
```