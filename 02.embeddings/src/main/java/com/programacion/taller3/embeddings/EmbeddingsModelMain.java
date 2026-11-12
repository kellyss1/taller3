package com.programacion.taller3.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.output.Response;

import java.nio.file.Paths;

public class EmbeddingsModelMain {
    static void main() {
        var pathModel = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\model.onnx");
        var pathToTokenizer = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\tokenizer.json");

        OnnxEmbeddingModel model = new OnnxEmbeddingModel(pathModel, pathToTokenizer, PoolingMode.MEAN);

        var text = "Hola, cómo estás?";

        Response<Embedding> response = model.embed(text);
        Embedding embedding = response.content();

        System.out.println("Dimension of the embedding: "+ embedding.dimension());
        System.out.println("Embedding vector: "+ embedding.vectorAsList());
    }
}
