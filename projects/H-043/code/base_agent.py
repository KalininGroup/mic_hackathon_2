from google import genai
from google.genai import types
from PIL import Image

from nanorange.settings import GOOGLE_API_KEY


class BaseImageAgent:
    """Base class for Gemini AI image processing agents."""
    
    def __init__(self, model: str, instruction: str) -> None:
        """
        Initialize the base image agent.
        
        Args:
            model: The Gemini model to use for processing
            instruction: The prompt/instruction for the model
        """
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = model
        self.instruction = instruction
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def process_tile(self, tile_path: str, output_path: str) -> bool:
        """
        Process a single tile with Gemini AI.
        
        Args:
            tile_path: Path to the input tile
            output_path: Path to save the processed tile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = Image.open(tile_path)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.instruction, image],
            )
            
            self._update_token_usage(response)
            
            saved = False
            for part in response.parts:
                if part.inline_data:
                    out = part.as_image()
                    out.save(output_path)
                    saved = True
                    break
            
            if not saved:
                print(f"⚠️  No image returned for {tile_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {tile_path}: {str(e)}")
            return False
    
    def _update_token_usage(self, response: types.GenerateContentResponse) -> None:
        """Update token usage statistics from response."""
        self.total_input_tokens += response.usage_metadata.prompt_token_count
        self.total_output_tokens += response.usage_metadata.candidates_token_count
    
    def print_usage_stats(self) -> None:
        """Print total token usage statistics."""
        print("\n" + "=" * 50)
        print("TOKEN USAGE SUMMARY")
        print("=" * 50)
        print(f"  Total input tokens:  {self.total_input_tokens:,}")
        print(f"  Total output tokens: {self.total_output_tokens:,}")
        print(f"  Total tokens:        {self.total_input_tokens + self.total_output_tokens:,}")
        print("=" * 50)

