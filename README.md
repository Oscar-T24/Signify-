### Structure of the CSV : 

`Frame`: id of the frame in the form `a`-`b` where `a` is the frame number and is unique for every recording (sequence of frame or unique frame), and `b` is the frame sub count. 
> For a sequence of frames, `b` gives the number of the jth frame within the `a`sequence
> for a single snapshot, `b` = 0 


`Landmark_ID` : id of the node (see picture from media pipe) identifies part on the hand

`Normalized_X` `Normalized_Y` are the normalized positions of each node (-1,1 on the x-y plane)


<img width="1392" alt="Screenshot 2025-02-02 at 13 18 32" src="https://github.com/user-attachments/assets/2ea900c8-1854-429c-9125-95ba9a8c1ff3" />

<img width="1392" alt="Screenshot 2025-02-02 at 13 20 21" src="https://github.com/user-attachments/assets/e54e4e1a-9b5f-4af6-b47f-4ef20aa3ec97" />

<img width="1392" alt="Screenshot 2025-02-02 at 13 21 20" src="https://github.com/user-attachments/assets/4342eacc-7e29-4675-a8d0-69079723927e" />
