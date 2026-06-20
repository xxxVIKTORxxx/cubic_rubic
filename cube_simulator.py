import random
import copy

class RubiksCubeSimulator:
    COLORS = ['W', 'Y', 'G', 'B', 'O', 'R']
    SIDES = ['Up', 'Down', 'Forward', 'Backward', 'Left', 'Right']

    def __init__(self):
        """Initializes the cube strictly in a SOLVED state."""
        self.sides = {}
        self.reset_to_solved()

    def reset_to_solved(self):
        """Sets every side to a uniform 3x3 layout of a single unique color."""
        for side, color in zip(self.SIDES, self.COLORS):
            self.sides[side] = [[color] * 3 for _ in range(3)]

    def scramble(self, steps=20):
        """Independent scrambler. Moves away from the current state."""
        history = []
        for _ in range(steps):
            side = random.choice(self.SIDES)
            direction = random.choice(['left', 'right'])
            self.move(side, direction)
            history.append((side, direction))
        return history

    def is_solved(self) -> bool:
        """User Win Checker: Returns True if all 6 sides are uniform."""
        for side in self.SIDES:
            matrix = self.sides[side]
            first_sticker = matrix[0][0]
            # Ensure every single sticker on this face matches the first one
            if not all(sticker == first_sticker for row in matrix for sticker in row):
                return False
        return True

    def display(self):
        """Prints the compact side-by-side or stacked grid view."""
        for side in self.SIDES:
            print(f"[{side}]")
            for row in self.sides[side]:
                print(" ".join(row))
        print("-" * 15)

    def _rotate_face_surface(self, side, direction):
        matrix = self.sides[side]
        if direction == 'right':
            # Clockwise rotation
            self.sides[side] = [list(x) for x in zip(*matrix[::-1])]
        else:
            # Counter-Clockwise rotation: Wrap the zip in a list() before slicing
            self.sides[side] = [list(x) for x in list(zip(*matrix))[::-1]]

    def move(self, side, direction):
        """
        Geometrically precise 3D move mechanics for a Rubik's Cube.
        """
        self._rotate_face_surface(side, direction)
        s = copy.deepcopy(self.sides)

        if side == 'Up':
            if direction == 'right':
                self.sides['Forward'][0]  = s['Right'][0]
                self.sides['Right'][0]    = s['Backward'][0]
                self.sides['Backward'][0] = s['Left'][0]
                self.sides['Left'][0]     = s['Forward'][0]
            else:
                self.sides['Forward'][0]  = s['Left'][0]
                self.sides['Left'][0]     = s['Backward'][0]
                self.sides['Backward'][0] = s['Right'][0]
                self.sides['Right'][0]    = s['Forward'][0]

        elif side == 'Down':
            if direction == 'right':
                self.sides['Forward'][2]  = s['Left'][2]
                self.sides['Left'][2]     = s['Backward'][2]
                self.sides['Backward'][2] = s['Right'][2]
                self.sides['Right'][2]    = s['Forward'][2]
            else:
                self.sides['Forward'][2]  = s['Right'][2]
                self.sides['Right'][2]    = s['Backward'][2]
                self.sides['Backward'][2] = s['Left'][2]
                self.sides['Left'][2]     = s['Forward'][2]

        elif side == 'Right':
            if direction == 'right':
                for i in range(3):
                    self.sides['Forward'][i][2]  = s['Down'][i][2]
                    self.sides['Down'][i][2]     = s['Backward'][2-i][0]
                    self.sides['Backward'][2-i][0] = s['Up'][i][2]
                    self.sides['Up'][i][2]       = s['Forward'][i][2]
            else:
                for i in range(3):
                    self.sides['Forward'][i][2]  = s['Up'][i][2]
                    self.sides['Up'][i][2]       = s['Backward'][2-i][0]
                    self.sides['Backward'][2-i][0] = s['Down'][i][2]
                    self.sides['Down'][i][2]     = s['Forward'][i][2]

        elif side == 'Left':
            if direction == 'right':
                for i in range(3):
                    self.sides['Forward'][i][0]  = s['Up'][i][0]
                    self.sides['Up'][i][0]       = s['Backward'][2-i][2]
                    self.sides['Backward'][2-i][2] = s['Down'][i][0]
                    self.sides['Down'][i][0]     = s['Forward'][i][0]
            else:
                for i in range(3):
                    self.sides['Forward'][i][0]  = s['Down'][i][0]
                    self.sides['Down'][i][0]     = s['Backward'][2-i][2]
                    self.sides['Backward'][2-i][2] = s['Up'][i][0]
                    self.sides['Up'][i][0]       = s['Forward'][i][0]

        elif side == 'Forward':
            if direction == 'right':
                for i in range(3):
                    self.sides['Up'][2][i]       = s['Left'][2-i][2]
                    self.sides['Right'][i][0]    = s['Up'][2][i]
                    self.sides['Down'][0][2-i]   = s['Right'][i][0]
                    self.sides['Left'][2-i][2]   = s['Down'][0][2-i]
            else:
                for i in range(3):
                    self.sides['Up'][2][i]       = s['Right'][i][0]
                    self.sides['Left'][2-i][2]   = s['Up'][2][i]
                    self.sides['Down'][0][2-i]   = s['Left'][2-i][2]
                    self.sides['Right'][i][0]    = s['Down'][0][2-i]

        elif side == 'Backward':
            if direction == 'right':
                for i in range(3):
                    self.sides['Up'][0][i]       = s['Right'][i][2]
                    self.sides['Left'][2-i][0]   = s['Up'][0][i]
                    self.sides['Down'][2][2-i]   = s['Left'][2-i][0]
                    self.sides['Right'][i][2]    = s['Down'][2][2-i]
            else:
                for i in range(3):
                    self.sides['Up'][0][i]       = s['Left'][2-i][0]
                    self.sides['Right'][i][2]    = s['Up'][0][i]
                    self.sides['Down'][2][2-i]   = s['Right'][i][2]
                    self.sides['Left'][2-i][0]   = s['Down'][2][2-i]