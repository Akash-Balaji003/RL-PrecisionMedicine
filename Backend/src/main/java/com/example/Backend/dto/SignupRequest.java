package com.example.Backend.dto;

import lombok.Data;

@Data
public class SignupRequest {
    private String username;
    private String password;
    private String name;
    private Integer age;
    private String gender;
}
