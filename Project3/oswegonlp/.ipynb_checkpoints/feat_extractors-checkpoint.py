import torch

class SimpleFeatureExtractor:

    def get_features(self, parser_state):
        """
        Take in all the parser state information and return features.
        Your features should be autograd.Variable objects of embeddings.

        :param parser_state the ParserState object for the current parse (giving access
            to the stack and input buffer)
        :return A list of autograd.Variable objects, which are the embeddings of your
            features
        """
        
        # Retrieve the top element from the stack
        top_of_stack = parser_state.stack_peek_n(1)[0]
        
        # Retrieve the first and second elements from the input buffer
        first_in_buffer, second_in_buffer = parser_state.input_buffer_peek_n(2)
        
        # Return embeddings of the top of stack, and first two elements of the input buffer
        return [top_of_stack.embedding, first_in_buffer.embedding, second_in_buffer.embedding]
