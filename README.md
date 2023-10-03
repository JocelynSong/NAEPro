<h1>Code, Model and Data for NAEPro</h1>

![image](https://github.com/JocelynSong/NAEPro/raw/main/NAEPro_model.png)


<h2>Installation</h2>
The dependencies can be set up using the following commands:

```ruby
conda create -n naepro python=3.8 -y 
conda activate naepro 
conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y 
bash setup.sh 
```

<h2>Training</h2>
Taking myoglobin as an example, we can train the model via the following scripts:

```ruby
bash train.sh
```

<h2>Inference</h2>
Taking myoglobin as an example, we can design myoglobins via the following scripts:

```ruby
bash design.sh
```

There are five items in the output directory:

1. protein.txt refers to the designed protein sequence
2. src.seq.txt refers to the ground truth sequences
3. pdb.txt refers to the target PDB ID and the corresponding chains
4. pred_pdbs refers to the directory of designed pdbs
5. tgt_pdbs refers to the directory of target pdbs



