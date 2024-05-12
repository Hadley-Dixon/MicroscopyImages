# MicroscopyImages
Final Project MATH 373

## Assignment Description
For the final project, your task is to:

Compete in the "Counting cells in microscopy imagesLinks to an external site." Kaggle contest that I've created for our class. You should implement a U-Net in PyTorch and train it using the data provided by the Kaggle contest. Use your trained U-Net to compete in the Kaggle competition. 
Write a short report telling the story of your work on this contest. What did you try? What did you get stuck on? How did you get unstuck? In the end, what approach yielded best results? You should include some examples of your results (segmented images) in your report. You should also include some convergence plots (that is, plots of cost function value vs. epoch) that you generated after training your model. Your report should be clearly, carefully written and well organized. It should be something that you could imagine hypothetically sharing with a manager or coworkers at your job. Your (hypothetical) coworkers should be impressed by the quality of your work. Your report should be clear and concise.
Your final Kaggle score for this contest must be less than 3. (In this contest, a lower score is better.) There will be a 25% penalty on your project grade if your final Kaggle score for this contest is greater than 3.

All code must be written yourself. Copying a U-Net implementation from the internet or from ChatGPT is considered academic dishonesty. You are allowed to get help from each other or from me, of course. It's ok to discuss strategy or approaches with classmates, as long as you're not copying someone else's code.

The original famous paper from 2015 which introduced the U-Net architecture can be found hereLinks to an external site..

You should get creative and try to find ways to improve your Kaggle score. For example, how many epochs of training gives you the best result? Which optimization algorithm and learning rate gives the best results? Try modifying the U-Net architecture and see if that helps. For example, you could try using just 3 downward blocks instead of 4. You could try using a different number of convolutions in each block, or different convolution filter sizes. Although the U-Net architecture is extremely popular, it seems unlikely that the generic U-Net happens to be optimal for our particular dataset.

Each time you train your U-Net, be sure to generate some "convergence plots", that is, plots of objective function value vs. epoch for both the training and validation datasets. (So, there will be two curves on one figure -- one curve corresponding to the training dataset, and one curve corresponding to the validation dataset.) We've been making such plots throughout the semester. Does the convergence plot reveal that overfitting begins after a certain number of epochs? Did you find early stopping to be helpful?

Submit your code along with your report.
