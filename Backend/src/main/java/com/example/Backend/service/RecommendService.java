package com.example.Backend.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class RecommendService {

    private final WebClient webClient = WebClient.create("http://localhost:8000");

    public String recommend(Map<String, Object> requestData) {
        // Call external API synchronously and extract "RecommendedTreatment"
        Mono<Map> responseMono = webClient.post()
                .uri("/recommend")
                .bodyValue(requestData)
                .retrieve()
                .bodyToMono(Map.class);

        Map<String, Object> response = responseMono.block();

        if (response != null && response.containsKey("RecommendedTreatment")) {
            return (String) response.get("RecommendedTreatment");
        } else {
            return "No recommendation available";
        }
    }
}
