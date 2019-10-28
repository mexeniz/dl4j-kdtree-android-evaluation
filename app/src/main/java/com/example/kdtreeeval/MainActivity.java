package com.example.kdtreeeval;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import org.deeplearning4j.clustering.kdtree.KDTree;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Eval";

    public static INDArray generateRandomVector(int arrayLength) {
        float[] randomFloats = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            // Generate a value between -1.0 and 1.0
            randomFloats[i] = (float) ((Math.random() * 2) - 1.0);
        }
        INDArray vector = Nd4j.create(Nd4j.createBuffer(randomFloats));

        return vector;
    }
    public void runKDTreeEval(){
        final int VECTOR_LENGTH = 512;
        final int TOTAL_VECTOR = 10000;
        KDTree tree = new KDTree(VECTOR_LENGTH);

        // Insert samples
        for (int i = 0; i < TOTAL_VECTOR; i++) {
            INDArray randomVec = generateRandomVector(VECTOR_LENGTH);
            INDArray unitVec = Transforms.unitVec(randomVec);

            tree.insert(unitVec);
        }
        Log.i(TAG, "finish init kdtree");

        long start = System.currentTimeMillis();
        INDArray targetVec = Transforms.unitVec(generateRandomVector(VECTOR_LENGTH));
        tree.knn(targetVec, 2.0);
        long end = System.currentTimeMillis();
        Log.i(TAG, String.format("knn cost_time=%.3f s", (end - start) / 1000.0));

        start = System.currentTimeMillis();
        targetVec = Transforms.unitVec(generateRandomVector(VECTOR_LENGTH));
        tree.knn(targetVec, 1.5);
        end = System.currentTimeMillis();
        Log.i(TAG, String.format("knn cost_time=%.3f s", (end - start) / 1000.0));

        start = System.currentTimeMillis();
        targetVec = Transforms.unitVec(generateRandomVector(VECTOR_LENGTH));
        tree.knn(targetVec, 1.0);
        end = System.currentTimeMillis();
        Log.i(TAG, String.format("knn cost_time=%.3f s", (end - start) / 1000.0));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        runKDTreeEval();
    }
}
