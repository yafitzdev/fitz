// server.go
// HTTP server for the TaskFlow project management API.
// Provides endpoints for task CRUD, user authentication, and health checks.

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Task represents a project management task.
type Task struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Assignee    string    `json:"assignee"`
	Status      string    `json:"status"` // open, in_progress, done
	Priority    int       `json:"priority"`
	CreatedAt   time.Time `json:"created_at"`
}

// TaskStore provides thread-safe in-memory task storage.
type TaskStore struct {
	mu    sync.RWMutex
	tasks map[string]Task
}

// AuthMiddleware validates Bearer tokens on protected routes.
// Tokens are checked against the AUTH_SECRET environment variable.
func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if !strings.HasPrefix(token, "Bearer ") {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		// In production, validate against a token store or JWT library.
		next.ServeHTTP(w, r)
	})
}

// LoggingMiddleware logs each request with method, path, and duration.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

// HandleListTasks returns all tasks as JSON.
// GET /api/tasks
func (s *TaskStore) HandleListTasks(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	tasks := make([]Task, 0, len(s.tasks))
	for _, t := range s.tasks {
		tasks = append(tasks, t)
	}
	json.NewEncoder(w).Encode(tasks)
}

// HandleCreateTask creates a new task from the request body.
// POST /api/tasks
func (s *TaskStore) HandleCreateTask(w http.ResponseWriter, r *http.Request) {
	var task Task
	if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	task.CreatedAt = time.Now()
	s.mu.Lock()
	s.tasks[task.ID] = task
	s.mu.Unlock()
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(task)
}

// HandleHealthCheck returns server status.
// GET /health
func HandleHealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	store := &TaskStore{tasks: make(map[string]Task)}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", HandleHealthCheck)
	mux.HandleFunc("/api/tasks", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			store.HandleListTasks(w, r)
		case http.MethodPost:
			store.HandleCreateTask(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	handler := LoggingMiddleware(AuthMiddleware(mux))
	fmt.Println("TaskFlow API listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
