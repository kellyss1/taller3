package com.programacion.taller3.embeddings;

import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.nio.file.Paths;

public class SimilitudMain {
    static void main() {
        var pathModel = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\model.onnx");
        var pathToTokenizer = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\tokenizer.json");

        OnnxEmbeddingModel model = new OnnxEmbeddingModel(pathModel, pathToTokenizer, PoolingMode.MEAN);

        var embedding1 = model.embed("Cual es el área de un círculo?").content();
        var embedding2 = model.embed("Cual es el área de un cuadrado?").content();
        var embedding3 = model.embed("Qué hora es?").content();

        //Calcular similitud entre embeddings
        double sim12 = CosineSimilarity.between(embedding1, embedding2);
        double sim13 = CosineSimilarity.between(embedding1, embedding3);
        double sim23 = CosineSimilarity.between(embedding2, embedding3);

        System.out.println("Similitud entre embedding 1 y 2: "+ sim12);
        System.out.println("Similitud entre embedding 1 y 3: "+ sim13);
        System.out.println("Similitud entre embedding 2 y 3: "+ sim23);
    }
}
