// UserService.java
// Service layer for user management in the Nexus Platform.
// Handles CRUD operations, password hashing, and role assignment.

package com.nexus.platform.service;

import com.nexus.platform.model.User;
import com.nexus.platform.model.Role;
import com.nexus.platform.repository.UserRepository;
import com.nexus.platform.exception.UserNotFoundException;
import com.nexus.platform.exception.DuplicateEmailException;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * UserService provides user management operations for the Nexus Platform.
 * All write operations are transactional.
 */
@Service
@Transactional
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Create a new user account.
     * Throws DuplicateEmailException if the email is already registered.
     */
    public User createUser(String email, String name, String rawPassword) {
        if (userRepository.findByEmail(email).isPresent()) {
            throw new DuplicateEmailException("Email already registered: " + email);
        }
        User user = new User();
        user.setEmail(email);
        user.setName(name);
        user.setPasswordHash(passwordEncoder.encode(rawPassword));
        user.setRole(Role.MEMBER);
        user.setCreatedAt(LocalDateTime.now());
        user.setActive(true);
        return userRepository.save(user);
    }

    /**
     * Retrieve a user by ID.
     * Throws UserNotFoundException if the user does not exist.
     */
    @Transactional(readOnly = true)
    public User getUserById(Long userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new UserNotFoundException("User not found: " + userId));
    }

    /**
     * List all active users, ordered by creation date descending.
     */
    @Transactional(readOnly = true)
    public List<User> listActiveUsers() {
        return userRepository.findAllByActiveOrderByCreatedAtDesc(true);
    }

    /**
     * Update a user's profile (name and email).
     * Throws UserNotFoundException if the user does not exist.
     * Throws DuplicateEmailException if the new email conflicts.
     */
    public User updateUser(Long userId, String newName, String newEmail) {
        User user = getUserById(userId);
        if (!user.getEmail().equals(newEmail)) {
            Optional<User> existing = userRepository.findByEmail(newEmail);
            if (existing.isPresent()) {
                throw new DuplicateEmailException("Email already in use: " + newEmail);
            }
        }
        user.setName(newName);
        user.setEmail(newEmail);
        return userRepository.save(user);
    }

    /**
     * Deactivate a user account (soft delete).
     * Throws UserNotFoundException if the user does not exist.
     */
    public void deactivateUser(Long userId) {
        User user = getUserById(userId);
        user.setActive(false);
        userRepository.save(user);
    }

    /**
     * Assign a new role to a user.
     * Valid roles: MEMBER, ADMIN, OWNER.
     * Throws UserNotFoundException if the user does not exist.
     * Throws IllegalArgumentException if the role is invalid.
     */
    public User assignRole(Long userId, Role newRole) {
        if (newRole == null) {
            throw new IllegalArgumentException("Role cannot be null");
        }
        User user = getUserById(userId);
        user.setRole(newRole);
        return userRepository.save(user);
    }
}
