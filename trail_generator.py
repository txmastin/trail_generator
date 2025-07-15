import pygame
import numpy as np
import sys

# --- Configuration ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SIM_AREA_SIZE = 512 # The area for the grid rendering
FPS = 30

# --- Colors ---
COLOR_BG = (25, 25, 40)
COLOR_GRID_BG = (40, 40, 60)
COLOR_TEXT = (220, 220, 220)
COLOR_INPUT_ACTIVE = (245, 158, 11)
COLOR_INPUT_INACTIVE = (100, 100, 120)
COLOR_BUTTON = (70, 70, 90)
COLOR_BUTTON_HOVER = (100, 100, 120)
COLOR_AGENT = (239, 68, 68)
COLOR_TRAIL = (245, 158, 11)


class InputBox:
    """A simple class to handle text input boxes in Pygame."""
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INPUT_INACTIVE
        self.text = text
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = COLOR_INPUT_ACTIVE if self.active else COLOR_INPUT_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                    self.color = COLOR_INPUT_INACTIVE
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def draw(self, screen, font):
        txt_surface = font.render(self.text, True, COLOR_TEXT)
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)


class TrailGenerator:
    """The core engine for generating the trail."""
    def __init__(self, grid_size, turn_prob, sparsity):
        self.size = grid_size
        self.turn_prob = turn_prob
        # Sparsity is now the same as forget_probability
        self.forget_prob = sparsity
        self.np_random = np.random.default_rng()

        self._direction_vectors = {
            0: np.array([-1, 0]), 1: np.array([0, 1]),
            2: np.array([1, 0]), 3: np.array([0, -1]),
        }
        self._move_to_direction_change = {"straight": 0, "left": -1, "right": 1}
        
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self._agent_location = self.np_random.integers(
            self.size // 2 - 2, self.size // 2 + 2, size=2, dtype=int
        )
        self._agent_direction = self.np_random.integers(0, 4)
        self.is_trapped = False

    def _is_move_valid(self, target_loc, current_loc, grid):
        if grid[target_loc[0], target_loc[1]] != 0:
            return False
        
        for i in range(4):
            check_loc = target_loc + self._direction_vectors[i]
            
            # It's okay for a neighbor to be the agent's current position.
            if np.array_equal(check_loc, current_loc):
                continue
            
            # If the neighbor is outside the grid, we don't need to check it.
            if not (0 <= check_loc[0] < self.size and 0 <= check_loc[1] < self.size):
                continue

            # If any other cardinal neighbor has a trail, the move is invalid.
            if grid[check_loc[0], check_loc[1]] != 0:
                return False
        return True

    def step(self):
        if self.is_trapped:
            return

        if self.np_random.random() > self.forget_prob:
            self.grid[self._agent_location[0], self._agent_location[1]] = 1

        valid_moves = {}
        for move, d_change in self._move_to_direction_change.items():
            new_dir = (self._agent_direction + d_change) % 4
            next_loc = self._agent_location + self._direction_vectors[new_dir]
            
            # Boundary check. If move is outside the grid, it's invalid.
            if not (0 <= next_loc[0] < self.size and 0 <= next_loc[1] < self.size):
                continue

            if self._is_move_valid(next_loc, self._agent_location, self.grid):
                valid_moves[move] = (next_loc, new_dir)

        if not valid_moves:
            self.is_trapped = True
            return

        can_straight = "straight" in valid_moves
        turns = [m for m in ["left", "right"] if m in valid_moves]
        can_turn = len(turns) > 0
        
        chosen_move = None
        if can_straight and can_turn:
            chosen_move = self.np_random.choice(turns) if self.np_random.random() < self.turn_prob else "straight"
        elif can_turn:
            chosen_move = self.np_random.choice(turns)
        else:
            chosen_move = "straight"

        self._agent_location, self._agent_direction = valid_moves[chosen_move]


class App:
    """The main application class that manages states, UI, and the simulation."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Trail Generation Tool")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        self.state = "INPUT" # INPUT, SIMULATING, FINISHED
        self.trail_engine = None
        self.trail_name = ""
        self.last_params = {} # Store last used parameters
        self.step_count = 0
        self.max_steps = 0
        self._setup_input_ui()

    def _setup_input_ui(self):
        # Use last parameters if they exist, otherwise start blank
        self.input_boxes = {
            "name": InputBox(300, 150, 300, 32, self.last_params.get("name", "")),
            "size": InputBox(300, 200, 140, 32, self.last_params.get("size", "")),
            "tortuosity": InputBox(300, 250, 140, 32, self.last_params.get("tortuosity", "")),
            "sparsity": InputBox(300, 300, 140, 32, self.last_params.get("sparsity", "")),
            "length": InputBox(300, 350, 140, 32, self.last_params.get("length", "0")),
        }
        self.generate_button_rect = pygame.Rect(300, 400, 200, 40)

    def _setup_finished_ui(self):
        self.save_button_rect = pygame.Rect(WINDOW_WIDTH - 220, 100, 200, 40)
        self.restart_button_rect = pygame.Rect(WINDOW_WIDTH - 220, 150, 200, 40)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if self.state == "INPUT":
                    self.handle_input_events(event)
                elif self.state == "FINISHED":
                    self.handle_finished_events(event)

            if self.state == "SIMULATING":
                self.trail_engine.step()
                self.step_count += 1
                
                # Check for termination conditions
                is_finished = self.trail_engine.is_trapped or \
                              (self.max_steps > 0 and self.step_count >= self.max_steps)
                
                if is_finished:
                    self.state = "FINISHED"
                    self._setup_finished_ui()

            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

    def handle_input_events(self, event):
        for box in self.input_boxes.values():
            box.handle_event(event)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.generate_button_rect.collidepoint(event.pos):
                self.start_simulation()

    def handle_finished_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.save_button_rect.collidepoint(event.pos):
                self.save_trail()
            elif self.restart_button_rect.collidepoint(event.pos):
                self.state = "INPUT"
                self._setup_input_ui() # This will now use the stored params

    def start_simulation(self):
        try:
            # Store current inputs to use for a potential restart
            self.last_params = {
                "name": self.input_boxes["name"].text,
                "size": self.input_boxes["size"].text,
                "tortuosity": self.input_boxes["tortuosity"].text,
                "sparsity": self.input_boxes["sparsity"].text,
                "length": self.input_boxes["length"].text,
            }
            
            self.trail_name = self.last_params["name"] or "untitled_trail"
            grid_size = int(self.last_params["size"])
            tortuosity = float(self.last_params["tortuosity"])
            sparsity = float(self.last_params["sparsity"])
            trail_length = int(self.last_params["length"])
            
            # Basic validation
            if not (0 <= tortuosity <= 1 and 0 <= sparsity <= 1 and grid_size > 0 and trail_length >= 0):
                print("Invalid input values. Please check ranges (0-1), size (>0), and length (>=0).")
                self.state = "INPUT" # Go back to input screen on error
                return
            
            self.max_steps = trail_length
            self.step_count = 0
            self.trail_engine = TrailGenerator(grid_size, tortuosity, sparsity)
            self.state = "SIMULATING"
        except (ValueError, TypeError) as e:
            print(f"Error parsing input: {e}. Please ensure numbers are entered correctly.")
            self.state = "INPUT" # Go back to input screen on error

    def save_trail(self):
        if not self.trail_engine:
            return
        
        filename = f"{self.trail_name}.txt"
        coordinates = []
        for r in range(self.trail_engine.size):
            for c in range(self.trail_engine.size):
                if self.trail_engine.grid[r, c] == 1:
                    # Save as (x, y) which corresponds to (column, row)
                    coordinates.append(f"({c}, {r})\n")
        
        try:
            with open(filename, "w") as f:
                f.writelines(coordinates)
            print(f"Trail saved successfully to {filename}")
        except IOError as e:
            print(f"Error saving file: {e}")


    def draw(self):
        self.screen.fill(COLOR_BG)
        if self.state == "INPUT":
            self.draw_input_screen()
        elif self.state == "SIMULATING" or self.state == "FINISHED":
            self.draw_simulation_screen()
            if self.state == "FINISHED":
                self.draw_finished_screen()
        pygame.display.flip()

    def draw_input_screen(self):
        # Title
        title_surf = self.font_large.render("Trail Generator Settings", True, COLOR_TEXT)
        self.screen.blit(title_surf, (WINDOW_WIDTH // 2 - title_surf.get_width() // 2, 80))

        # Labels and boxes
        labels = {
            "name": "Trail Name:", 
            "size": "Grid Size:", 
            "tortuosity": "Tortuosity (0-1):", 
            "sparsity": "Sparsity (0-1):",
            "length": "Trail Length (0=max):"
        }
        for name, box in self.input_boxes.items():
            label_surf = self.font_small.render(labels[name], True, COLOR_TEXT)
            self.screen.blit(label_surf, (box.rect.x - label_surf.get_width() - 10, box.rect.y + 5))
            box.draw(self.screen, self.font_large)

        # Generate Button
        mouse_pos = pygame.mouse.get_pos()
        button_color = COLOR_BUTTON_HOVER if self.generate_button_rect.collidepoint(mouse_pos) else COLOR_BUTTON
        pygame.draw.rect(self.screen, button_color, self.generate_button_rect, border_radius=5)
        btn_text = self.font_large.render("Generate Trail", True, COLOR_TEXT)
        self.screen.blit(btn_text, (self.generate_button_rect.centerx - btn_text.get_width() // 2, self.generate_button_rect.centery - btn_text.get_height() // 2))

    def draw_simulation_screen(self):
        # Draw grid background
        grid_rect = pygame.Rect((WINDOW_WIDTH - SIM_AREA_SIZE) // 2, (WINDOW_HEIGHT - SIM_AREA_SIZE) // 2, SIM_AREA_SIZE, SIM_AREA_SIZE)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, grid_rect)
        
        engine = self.trail_engine
        pix_size = SIM_AREA_SIZE / engine.size

        # Draw trail and agent
        for r in range(engine.size):
            for c in range(engine.size):
                if engine.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, COLOR_TRAIL, pygame.Rect(grid_rect.x + c * pix_size, grid_rect.y + r * pix_size, pix_size, pix_size))
        
        # Draw agent on top
        agent_x = grid_rect.x + (engine._agent_location[1] + 0.5) * pix_size
        agent_y = grid_rect.y + (engine._agent_location[0] + 0.5) * pix_size
        pygame.draw.circle(self.screen, COLOR_AGENT, (agent_x, agent_y), pix_size / 2)

    def draw_finished_screen(self):
        # Overlay with options
        overlay = pygame.Surface((250, 200), pygame.SRCALPHA)
        overlay.fill((40, 40, 60, 220))
        self.screen.blit(overlay, (WINDOW_WIDTH - 260, 50))

        # Title
        title_surf = self.font_large.render("Generation Finished", True, COLOR_TEXT)
        self.screen.blit(title_surf, (WINDOW_WIDTH - 250, 60))

        # Save Button
        mouse_pos = pygame.mouse.get_pos()
        save_color = COLOR_BUTTON_HOVER if self.save_button_rect.collidepoint(mouse_pos) else COLOR_BUTTON
        pygame.draw.rect(self.screen, save_color, self.save_button_rect, border_radius=5)
        save_text = self.font_large.render("Save Trail", True, COLOR_TEXT)
        self.screen.blit(save_text, (self.save_button_rect.centerx - save_text.get_width() // 2, self.save_button_rect.centery - save_text.get_height() // 2))

        # Restart Button
        restart_color = COLOR_BUTTON_HOVER if self.restart_button_rect.collidepoint(mouse_pos) else COLOR_BUTTON
        pygame.draw.rect(self.screen, restart_color, self.restart_button_rect, border_radius=5)
        restart_text = self.font_large.render("Restart", True, COLOR_TEXT)
        self.screen.blit(restart_text, (self.restart_button_rect.centerx - restart_text.get_width() // 2, self.restart_button_rect.centery - restart_text.get_height() // 2))


if __name__ == '__main__':
    app = App()
    app.run()

