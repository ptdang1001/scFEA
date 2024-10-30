### V 0.51
 - Entirely new implementation of scFEA & MPO

### Instructions

1. Use of Server for Running the Model:  
   The new version introduces a more complex model, which I recommend running on a server rather than a local laptop to ensure optimal performance. Depending on your dataset, the runtime may be long. Please use the `sbatch run_server.sh` script (you may need to adjust it according to your specific server setup) to submit the job to the server.  

   GPU Use: Ideally, the model should run with at least one GPU for better performance. However, if a GPU is unavailable, the model can still run on a CPU, although it might take longer unless your sample size is small or the reaction network is very simple.  

2. Multiple Output Files:  
   After the job completes, you will find multiple output files generated in the output directories you specified.  I recommend "flux_scfea_mpo.csv".

3.  Virtual Environment Recommendation:
I recommend using Anaconda to create a virtual environment to avoid any library conflicts. This will ensure a smoother setup and execution of the code. There is a "requirements.txt" file to specify the python package version information.
4. Running Locally:
If your data is small, you may use the command "sh run_local.sh" to run the model on your local machine to make it easier. Don't forget to update some parameters like gene expression file name, input/output directory etc.

