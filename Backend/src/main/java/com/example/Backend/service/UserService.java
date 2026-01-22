package com.example.Backend.service;

import com.example.Backend.model.*;
import com.example.Backend.repository.*;
import com.example.Backend.dto.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final PatientRepository patientRepository;

    @Transactional
    public boolean signup(SignupRequest request) {
        if (userRepository.existsByUsername(request.getUsername())) {
            return false;
        }

        User user = User.builder()
                .username(request.getUsername())
                .password(request.getPassword())  // TODO: hash password here
                .build();

        userRepository.save(user);

        Patient patient = Patient.builder()
                .name(request.getName())
                .age(request.getAge())
                .gender(request.getGender())
                .build();

        patientRepository.save(patient);

        return true;
    }

    public boolean login(String username, String password) {
        User user = userRepository.findByUsername(username);
        if (user == null) return false;
        // TODO: Check hashed password here instead of plain text
        return user.getPassword().equals(password);
    }
}
