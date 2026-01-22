package com.example.Backend.controller;

import com.example.Backend.service.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/treatment")
@RequiredArgsConstructor
public class RecommendController {

    private final RecommendService recommendService;

    @PostMapping("/recommend")
    public ResponseEntity<String> recommend(@RequestBody Map<String, Object> requestData) {
        String recommendedTreatment = recommendService.recommend(requestData);
        return ResponseEntity.ok(recommendedTreatment);
    }
}
