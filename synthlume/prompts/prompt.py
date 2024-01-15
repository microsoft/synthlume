import re
import os
import pkg_resources

from synthlume.prompts.tags import Tag
from synthlume.logging.logging import get_logger
logger = get_logger(__name__)

class Prompt:
    def __init__(self, path: str) -> None:
        assert os.path.exists(path), f"Prompt file {path} does not exist"

        self.path = path
        with open(self.path, "r") as f:
            self.text = f.read()
        
        self.tags, self.text = self._get_prompt_tags()
        self.keys = self._find_prompt_keys()

    @classmethod
    def from_data(cls, language: str, name: str) -> "Prompt":
        resource_path = f"{language}/{name}.prompt"
        file_path = pkg_resources.resource_filename(__name__, resource_path)

        prompt = cls(file_path)
        return prompt

    def _find_prompt_keys(self) -> set[str]:
        pattern = r'(?<!\{)\{([^}]*)\}(?!\})'
        matches = list(re.findall(pattern, self.text))
        logger.debug(f"Found keys {matches} in prompt {self.path}")

        return set(matches)
    
    def _get_prompt_tags(self) -> list[str]:
        if "############" not in self.text:
            logger.debug(f"No tags found in prompt {self.path}")
            return [], self.text

        tags, clean_text = self.text.split("############")

        if not tags.startswith("TAGS:"):
            logger.warning(f"In prompt {self.path} found '############' but no tags, make sure to add 'TAGS:' before tags")
            return [], self.text
        
        pattern = r"#(\w+)"
        tags = list(re.findall(pattern, tags))

        valid_tags = {tag.value for tag in Tag}

        for tag in tags:
            if tag not in valid_tags:
                logger.warning(f"Found tag {tag} in prompt {self.path} but it is not a valid tag")

        logger.debug(f"Found tags {tags} in prompt {self.path}")

        return tags, clean_text