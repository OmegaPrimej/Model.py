import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Log

class NexusFloatHub(App):
    """A floating AI interface for Termux."""
    CSS = """
    Screen { layout: grid; grid-size: 2; grid-rows: 1fr 1fr; }
    #log-box { row-span: 1; border: double skyblue; }
    #ai-decision { row-span: 1; border: solid lime; }
    #input-box { column-span: 2; border: solid magenta; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Log(id="log-box")         # Box 1: Picks up incoming logging
        yield Static("AI DECIDING...", id="ai-decision")  # Box 2: Decision/Reflex
        yield Input(placeholder="Type to Chat / Real-time adapt", id="input-box")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles chat adaptation and blending."""
        user_text = event.value
        # Add logic here for 'reversal' or 'cross merge'
        adapted_text = user_text[::-1] # Example: Reflex backwards
        self.query_one("#log-box").write_line(f"User: {user_text}")
        self.query_one("#ai-decision").update(f"Adapted: {adapted_text}")
        event.input.value = ""

if __name__ == "__main__":
    NexusFloatHub().run()

