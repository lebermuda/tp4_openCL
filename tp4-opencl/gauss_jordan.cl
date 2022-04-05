__kernel void vecadd(__global int *k,__local Matric mat,__global float[] rowPivot,__global float[] rowFuturPivot,__gloabl int indexMax,__global struct localMax) { 
    // etape k, sous matrice des lignes traité par un kernel, row du pivot, row de la ligne à échanger avec celle du pivot, index du max, tableau de tous les max locaux

    int id = get_group_id(0);
    int n = get_global_size(0);
    int nKernel = 4;

    rowPivot/=rowPivot[k];
    
    if(k=!0){
        //Echange de ligne entre pivot et max
        if(id*n/nKernel<=k && id*(n/nKernel+1)>k){
            mat[k-id*n/nKernel]=rowPivot;
        }
        if(id*n/nKernel<=indexMax && id*(n/nKernel+1)>indexMax){
            mat[indexMax-id*n/nKernel]=rowFuturPivot;
        }

        //soustraction pour avoir une colonnne de 0
        for (int i=0;i<mat.size();i++){
            if(id*n/nKernel+i != k){
                mat[indexMax-id*n/nKernel] -= mat[indexMax-id*n/nKernel][k]/rowPivot[k]*rowPivot ;
            }
        }
    }

    //Trouver le max local
    localMax[id]=[mat[k][k],k,mat[k]];
    int i=k+1;
    while (i<n){
        if(mat[i-id*n/nKernel][k]>localMax[id][0]){
            localMax[id]=[mat[i-id*n/nKernel][k],i,mat[i-id*n/nKernel]]
        }
        i++;
    }

    //Envoie de la rowFuturPivot;
    if(id*n/nKernel<=k+1 && id*(n/nKernel+1)>k+1){
        rowFuturPivot=mat[k+1];
    }

    //Reste sur la CPU => trouver le max global
}