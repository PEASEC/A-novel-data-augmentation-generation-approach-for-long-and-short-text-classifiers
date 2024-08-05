def get_generated_examples(training_data, class_to_be_augmented):
	gpt2 = get_gpt2_model()
	class_data = get_specific_data(from=training_data, 
									for_class=class_to_be_augmented)
	
	# Safety step 1
	for instance in class_data:
		if instance is a News_Document:
			instance = attach(prefix="<|startoftext|> ", to=instance)
		else:
			instance = attach(prefix="<|startoftext|> " + 
							index_of_instance(instance), to=instance)
		instance = attach(suffix=" <|endoftext|>", to=instance)
		
	# Safety step 2
	for 2000-8000 steps:
		gpt2.fintune(with_examples_in=class_data)
		
	for instance in class_data:
		prefix = "<|startoftext|>"
		
		if instance is a News_Document:
			prefix += title_of(instance)
		else:
			prefix += index_of_instance(instance)
	
		generated_data = gpt2.generate(with_prefix=prefix, 
										number_of_examples=10, 
										temperature=[0.7, 0.8, 0.9])
		
	# Safety step 3
	sbert = get_sbert_model()
	ground_truth_embeddings = sbert.encode(class_data)
	generated_data_embeddings = sbert.encode(generated_data)
	
	centroid_of_ground_truth = get_centroid_of(embeddings=ground_truth_embeddings)
	distances = get_distances(from=generated_data_embeddings, 
								to=centroid_of_ground_truth, 
								with=CosineSimilarity)
								
	generated_data_embeddings = remove_examples(from=generated_data, 
												with=distances)
	
	return generated_data