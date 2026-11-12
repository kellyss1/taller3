package com.programacion.taller3.embeddings;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.nio.file.Paths;
import java.util.List;

public class BusquedaSemanticaMain {
    static void main() {

        var pathModel = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\model.onnx");
        var pathToTokenizer = Paths.get("C:\\Users\\Kellyss\\Documents\\Workspace\\taller3\\modelos\\multilingual-e5-small\\tokenizer.json");

        OnnxEmbeddingModel model = new OnnxEmbeddingModel(pathModel, pathToTokenizer, PoolingMode.MEAN);

        var documentos = List.of(
                "Java es un lenguaje de programación orientada a objetos",
                "Phyton es un lenguaje prpular de inteligencia artificial",
                "Los embeddings representan texto como vectores"
        );

        List<Embedding> embeddings = documentos.stream()
                .map(model::embed)
                .map(Response::content)
                .toList();

        //Consultar
        var consulta = "Programación orientada a objetos";
        var embeddingConsulta = model.embed(consulta).content();

        int index =0;
        int mejorIndice= 0;
        double mejorScore = -1.0;

        for (var it:embeddings) {
            double score = CosineSimilarity.between(embeddingConsulta, it);
            System.out.printf("Documento [%d] - Similitud: %.4f%n", index, score);

            if (score > mejorScore) {
                mejorScore = score;
                mejorIndice = index;
            }
            index++;
        }


        System.out.printf("Mejor resultado: %d - %s%n", mejorIndice, documentos.get(mejorIndice));
        System.out.printf("Similitud: %.4f%n", mejorScore);
    }
}
