# Symmetric Visual Force Learning

This repositroy contrains the code of the paper **Symmetric Models for Visual Force Policy Learning**

## Install
1. Install [BulletArm](https://github.com/ColinKohler/BulletArm/tree/v1.1) v1.1 (unreleased)
2. Clone this repository
```
git clone https://github.com/ColinKohler/SymmetricVisualForceLearning.git
```
3. Install dependencies from Pipfile
```
pipenv install
```
4. Create results directory to store train logs and model
```
mkdir SymmetricVisualForceLearning/data/
```

## Running SVFL
```
python scripts/train.py [task]
```

Replace [task] with the desired domain to train on. Currently avaliable tasks are: 
  * Block Picking: block_picking
  * Block Pulling: block_pulling
  * Block Pushing: block_pushing
  * Block Pulling Corner: block_pulling_corner
  * Mug Picking: mug_picking
  * Household Picking: clutter_picking
  * Drawer Opening: drawer_opening
  * Drawer Closing: drawer_closing
  * Peg Insertion: peg_insertion

For example, to run block picking:
```
python scripts/train.py block_picking
```
   
For additional information, you can examine the training script arguments in the normal manner:
```
python scripts/train.py --h
```

## Configs
The simulation configurations and hyperparameters can be found in ```configs/task.py```. 

## Cite

## Reference
